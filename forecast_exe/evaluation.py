"""Expanding chronological backtests and horizon/season diagnostics."""

import pandas as pd

from forecast_exe.model import scores, train, validate_split

SEASONS = {0: "Winter (DJF)", 1: "Spring (MAM)", 2: "Summer (JJA)", 3: "Autumn (SON)"}


def rolling_splits(index, folds=3, test_days=90, calibration_days=120):
    """Non-overlapping test-origin blocks with expanding fit and purged calibration."""
    if folds < 2 or folds > 8 or test_days < 30 or calibration_days < 60:
        raise ValueError("Use 2–8 folds, at least 30 test days and 60 calibration days")
    first = len(index) - folds * test_days
    if first - calibration_days - 14 < 365:
        raise ValueError("Insufficient history for this rolling setup; load at least three years")
    out = []
    for number in range(folds):
        start = first + number * test_days
        test = index[start : start + test_days]
        cal = index[start - 7 - calibration_days : start - 7]
        fit = index[: start - 14 - calibration_days]
        validate_split(fit, cal, test)
        out.append((fit, cal, test))
    return out


def rolling_backtest(df, meta, seed=42, folds=3, test_days=90, progress=None):
    """Refit each fold; each test is untouched by that fold's training/calibration."""
    splits = rolling_splits(df.index[29:-7], folds, test_days)
    metrics, records, boundaries = [], [], []
    for number, split in enumerate(splits):

        def update(value):
            if progress:
                progress((number + value) / len(splits))

        bundle = train(df, meta, seed, progress=update, split=split)
        metrics.append(bundle["metrics"].assign(fold=number + 1, city=meta["location"]["name"]))
        records.append(bundle["predictions"].assign(fold=number + 1, city=meta["location"]["name"]))
        boundaries.append(dict(fold=number + 1, **bundle["metadata"]["boundaries"]))
    return dict(
        metrics=pd.concat(metrics, ignore_index=True),
        predictions=pd.concat(records, ignore_index=True),
        boundaries=boundaries,
        synthetic=bool(meta.get("synthetic")),
    )


def seasonal_diagnostics(predictions):
    rows = []
    for (h, target, season), group in predictions.groupby(["horizon", "target", "season"]):
        for name, column in [
            ("ML", "predicted"),
            ("Persistence", "persistence"),
            ("Climatology", "climatology"),
        ]:
            values = scores(group.actual, group[column], target == "rain")
            if target != "rain" and name == "ML":
                values["coverage"] = float(
                    ((group.actual >= group.lower) & (group.actual <= group.upper)).mean()
                )
                values["width"] = float((group.upper - group.lower).mean())
            rows.append(dict(horizon=h, target=target, season=SEASONS[season], model=name, **values))
    return pd.DataFrame(rows)
