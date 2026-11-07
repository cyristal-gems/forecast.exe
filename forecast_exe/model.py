"""Direct horizon models with purged chronological evaluation and calibration."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

from forecast_exe.data import validate

VERSION = "2.0.0"
THRESHOLD = 0.1


def features(df, horizon):
    """Features at t use observations <= t; target calendar is known in advance."""
    result = pd.DataFrame(index=df.index)
    for col in ["high", "low", "rain"]:
        for lag in [0, 1, 2, 7, 14]:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
        for window in [3, 7, 14, 30]:
            result[f"{col}_mean{window}"] = df[col].rolling(window).mean()
    target = df.index + pd.Timedelta(days=horizon)
    result["sin"] = np.sin(2 * np.pi * target.strftime("%j").astype(int) / 365.25)
    result["cos"] = np.cos(2 * np.pi * target.strftime("%j").astype(int) / 365.25)
    return result


def partitions(index):
    """Seven-day purge prevents any train/calibration/test target overlap."""
    a, b = int(len(index) * 0.65), int(len(index) * 0.82)
    return index[: a - 7], index[a : b - 7], index[b:]


def scores(y, pred, rain=False):
    """Metrics with defined empty and one-class behavior."""
    if len(y) == 0:
        return {"n": 0}
    y, pred = np.asarray(y), np.asarray(pred)
    if rain:
        return dict(
            n=len(y),
            brier=float(brier_score_loss(y, pred)),
            auc=float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else None,
            confusion=confusion_matrix(y, pred >= 0.5, labels=[0, 1]).tolist(),
        )
    return dict(
        n=len(y),
        mae=float(np.mean(abs(y - pred))),
        rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
        bias=float(np.mean(pred - y)),
    )


def climatology(train, dates, target):
    """Circular +/-15 day seasonal window, fitted exclusively on training data."""
    values = (train.rain >= THRESHOLD).astype(float) if target == "rain" else train[target]
    doy = train.index.strftime("%j").astype(int).to_numpy()
    out = []
    for day in dates.strftime("%j").astype(int):
        distance = abs(doy - day)
        mask = np.minimum(distance, 366 - distance) <= 15
        out.append(values[mask].mean() if mask.any() else values.mean())
    return np.array(out)


def reconcile(high, low):
    """Reconcile crossing point estimates by ordering the two predicted extremes."""
    return np.maximum(high, low), np.minimum(high, low)


def train(df, provenance, seed=42, progress=None, split=None):
    """Fit 21 direct models; final test data never fit models or intervals."""
    validate(df)
    if len(df) < 730:
        raise ValueError("At least 730 complete daily observations are required")
    usable = df.index[29:-7]
    fit, cal, test = split if split is not None else partitions(usable)
    validate_split(fit, cal, test)
    all_x = {h: features(df, h) for h in range(1, 8)}
    models, intervals, rows, predictions = {}, {}, [], []
    calibrators, seasonal_intervals = {}, {}
    for h in range(1, 8):
        if progress:
            progress((h - 1) / 7)
        x = all_x[h]
        temp_cal, temp_test = {}, {}
        for target in ["high", "low", "rain"]:
            y = df[target].shift(-h)
            if target == "rain":
                y = (y >= THRESHOLD).astype(int)
                estimator = (
                    DummyClassifier(strategy="prior")
                    if y.loc[fit].nunique() < 2
                    else HistGradientBoostingClassifier(
                        max_iter=70, max_leaf_nodes=15, l2_regularization=2, random_state=seed
                    )
                )
            else:
                estimator = HistGradientBoostingRegressor(
                    max_iter=70, max_leaf_nodes=15, l2_regularization=2, random_state=seed
                )
            estimator.fit(x.loc[fit], y.loc[fit])
            models[h, target] = estimator
            if target == "rain":
                raw = probability(estimator, x.loc[test])
                calibrator = fit_calibrator(probability(estimator, x.loc[cal]), y.loc[cal])
                calibrators[h] = calibrator
                pred = calibrate(calibrator, raw)
            else:
                temp_cal[target] = estimator.predict(x.loc[cal])
                temp_test[target] = estimator.predict(x.loc[test])
                continue
            append_results(rows, predictions, df, fit, test, h, target, y.loc[test], pred, raw=raw)
            rows[-3]["raw_brier"] = scores(y.loc[test], raw, True)["brier"]
        temp_cal["high"], temp_cal["low"] = reconcile(temp_cal["high"], temp_cal["low"])
        temp_test["high"], temp_test["low"] = reconcile(temp_test["high"], temp_test["low"])
        for target in ["high", "low"]:
            y = df[target].shift(-h)
            residuals = y.loc[cal].to_numpy() - temp_cal[target]
            q = np.quantile(residuals, [0.1, 0.9])
            intervals[h, target] = q
            cal_seasons = seasons(cal + pd.Timedelta(days=h))
            for season in range(4):
                group = residuals[cal_seasons == season]
                seasonal_intervals[h, target, season] = (
                    np.quantile(group, [0.1, 0.9]) if len(group) >= 40 else q
                )
            quantiles = np.array(
                [seasonal_intervals[h, target, season] for season in seasons(test + pd.Timedelta(days=h))]
            )
            pred = temp_test[target]
            lower, upper = pred + quantiles[:, 0], pred + quantiles[:, 1]
            append_results(
                rows, predictions, df, fit, test, h, target, y.loc[test], pred, lower=lower, upper=upper
            )
            rows[-3]["coverage"] = float(np.mean((y.loc[test] >= lower) & (y.loc[test] <= upper)))
            rows[-3]["width"] = float(np.mean(upper - lower))
        if progress:
            progress(h / 7)
    boundaries = {
        name: [str(idx.min().date()), str(idx.max().date())]
        for name, idx in [("train_origins", fit), ("calibration_origins", cal), ("test_origins", test)]
    }
    return dict(
        version=VERSION,
        models=models,
        intervals=intervals,
        calibrators=calibrators,
        seasonal_intervals=seasonal_intervals,
        metrics=pd.DataFrame(rows),
        predictions=pd.DataFrame(predictions),
        feature_names=list(all_x[1]),
        metadata=dict(
            seed=seed,
            data_fingerprint=fingerprint(df),
            version=VERSION,
            packages={
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
            calibration={
                str(h): "sigmoid" if calibrators[h] is not None else "raw (insufficient class variation)"
                for h in range(1, 8)
            },
            sklearn=sklearn.__version__,
            provenance=provenance,
            boundaries=boundaries,
            training_end=str((fit.max() + pd.Timedelta(days=7)).date()),
            origin=str(df.index[-1].date()),
            history_start=str(df.index[0].date()),
            created_at=datetime.now(UTC).isoformat(),
            label="rain >= 0.1 mm; probability decision threshold 0.5",
            estimator="HistGradientBoosting (or constant prior for one-class rain)",
        ),
    )


def append_results(rows, records, df, fit, test, h, target, y, pred, lower=None, upper=None, raw=None):
    dates = test + pd.Timedelta(days=h)
    train_reference = df.loc[: fit.max() + pd.Timedelta(days=h)]
    persistence = (
        df.rain.ge(THRESHOLD).rolling(30).mean().loc[test].to_numpy()
        if target == "rain"
        else df.loc[test, target].to_numpy()
    )
    seasonal = climatology(train_reference, dates, target)
    for name, values in [("ML", pred), ("Persistence", persistence), ("Climatology", seasonal)]:
        metric = scores(y, values, target == "rain")
        rows.append(dict(horizon=h, target=target, model=name, **metric))
    for i, (date, actual, estimate) in enumerate(zip(dates, y, pred)):
        records.append(
            dict(
                date=date,
                horizon=h,
                target=target,
                actual=float(actual),
                predicted=float(estimate),
                persistence=float(persistence[i]),
                climatology=float(seasonal[i]),
                season=int(seasons(pd.DatetimeIndex([date]))[0]),
                lower=float(lower[i]) if lower is not None else np.nan,
                upper=float(upper[i]) if upper is not None else np.nan,
                raw=float(raw[i]) if raw is not None else np.nan,
            )
        )


def probability(model, x):
    if 1 not in model.classes_:
        return np.zeros(len(x))
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def forecast(bundle, df, location, demo=False, now=None):
    """Reject cross-location, changed history, incomplete or stale live origins."""
    validate(df)
    meta = bundle["metadata"]
    if bundle["version"] != VERSION or meta["sklearn"] != sklearn.__version__:
        raise ValueError("Incompatible artifact; retrain with this installation")
    if meta["provenance"]["location"] != location:
        raise ValueError("This model belongs to a different location; retrain")
    if bool(meta["provenance"].get("synthetic")) != demo:
        raise ValueError("Synthetic and real models cannot be interchanged")
    if str(df.index[-1].date()) != meta["origin"]:
        raise ValueError("Observation snapshot changed; explicitly retrain the model")
    if fingerprint(df) != meta.get("data_fingerprint"):
        raise ValueError(
            "Observation snapshot changed; dataset fingerprint differs. Retrain this experiment."
        )
    today = pd.Timestamp(now or datetime.now(ZoneInfo(location["timezone"]))).date()
    age = (today - df.index[-1].date()).days
    if not demo and age != 1:
        raise ValueError(
            f"Live forecast unavailable: last complete data {df.index[-1].date()} "
            f"is {age} days from today. Yesterday is required. Historical evaluation remains available."
        )
    result = []
    for h in range(1, 8):
        x = features(df, h).iloc[[-1]]
        if list(x) != bundle["feature_names"]:
            raise ValueError("Feature schema changed; retrain")
        high, low = reconcile(
            bundle["models"][h, "high"].predict(x)[0], bundle["models"][h, "low"].predict(x)[0]
        )
        row = dict(
            date=df.index[-1] + pd.Timedelta(days=h),
            horizon=h,
            high=float(high),
            low=float(low),
            rain=float(calibrate(bundle["calibrators"][h], probability(bundle["models"][h, "rain"], x))[0]),
        )
        for target in ["high", "low"]:
            q = bundle["seasonal_intervals"][h, target, int(seasons(pd.DatetimeIndex([row["date"]]))[0])]
            row[f"{target}_lower"], row[f"{target}_upper"] = row[target] + q[0], row[target] + q[1]
        row.update(
            city=location["name"],
            timezone=location["timezone"],
            origin=meta["origin"],
            model_version=bundle["version"],
            forecast_source="Locally trained ML",
            reference_source=meta["provenance"]["source"],
            synthetic=bool(meta["provenance"].get("synthetic")),
            temperature_unit="°C",
            rain_definition="Daily precipitation >= 0.1 mm",
            rain_probability_unit="fraction (0–1)",
        )
        result.append(row)
    return pd.DataFrame(result)


def save(bundle, path):
    """Write locally generated trusted artifacts and inspectable JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_bundle(bundle)
    temporary = path.with_suffix(".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(path)
    metadata_path = path.with_suffix(".json")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(json.dumps(bundle["metadata"], indent=2))
    metadata_temporary.replace(metadata_path)


def load(path):
    """Load only application-generated local files; never accept artifact uploads."""
    bundle = joblib.load(path)
    validate_bundle(bundle)
    return bundle


def fingerprint(df):
    """Content identity includes all dates, values, units schema and column names."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    digest.update(str(list(df.columns)).encode())
    return digest.hexdigest()


def seasons(index):
    """Calendar quarters of meteorological seasons: DJF, MAM, JJA, SON."""
    return np.asarray(index.month % 12 // 3)


def validate_split(fit, cal, test):
    if min(map(len, (fit, cal, test))) < 20:
        raise ValueError("Each chronological partition needs at least 20 origins")
    if fit.max() + pd.Timedelta(days=7) >= cal.min() or cal.max() + pd.Timedelta(days=7) >= test.min():
        raise ValueError("Split target windows overlap; purge at least seven origin days")


def fit_calibrator(probabilities, labels):
    """Fit sigmoid probability calibration solely on calibration-period outcomes."""
    probabilities, labels = np.asarray(probabilities), np.asarray(labels)
    if len(labels) < 30 or len(np.unique(labels)) < 2 or len(np.unique(probabilities)) < 2:
        return None
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1 - 1e-6))
    return LogisticRegression(C=1, random_state=42).fit(logits[:, None], labels)


def calibrate(calibrator, probabilities):
    probabilities = np.asarray(probabilities)
    if calibrator is None:
        return probabilities
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1 - 1e-6))
    return calibrator.predict_proba(logits[:, None])[:, 1]


def validate_bundle(bundle):
    """Validate the complete application schema after loading trusted local files."""
    required = {
        "version",
        "models",
        "metadata",
        "feature_names",
        "intervals",
        "metrics",
        "predictions",
        "calibrators",
        "seasonal_intervals",
    }
    try:
        if not isinstance(bundle, dict) or not required.issubset(bundle) or bundle["version"] != VERSION:
            raise ValueError("Incomplete or incompatible model; retrain")
        meta = bundle["metadata"]
        if meta["sklearn"] != sklearn.__version__ or not meta["data_fingerprint"] or not meta["provenance"]:
            raise ValueError("Model library mismatch or missing provenance; retrain")
        expected_features = list(
            features(pd.DataFrame(columns=["high", "low", "rain"], index=pd.DatetimeIndex([])), 1)
        )
        if bundle["feature_names"] != expected_features:
            raise ValueError("Feature schema changed; retrain")
        expected_models = {(h, t) for h in range(1, 8) for t in ["high", "low", "rain"]}
        if set(bundle["models"]) != expected_models:
            raise ValueError("Incomplete horizon models; retrain")
        for h in range(1, 8):
            for target in ["high", "low", "rain"]:
                if not hasattr(bundle["models"][h, target], "predict"):
                    raise ValueError("Incomplete estimator; retrain")
            for target in ["high", "low"]:
                for season in range(4):
                    q = np.asarray(bundle["seasonal_intervals"][h, target, season])
                    if q.shape != (2,) or not np.isfinite(q).all() or q[0] > q[1]:
                        raise ValueError("Invalid calibration intervals; retrain")
            if h not in bundle["calibrators"]:
                raise ValueError("Missing probability calibration; retrain")
            calibrator = bundle["calibrators"][h]
            if calibrator is not None and not hasattr(calibrator, "predict_proba"):
                raise ValueError("Invalid probability calibration; retrain")
        if len(bundle["metrics"]) != 63 or bundle["predictions"].empty:
            raise ValueError("Missing evaluation; retrain")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Incomplete model artifact; retrain this location") from exc
