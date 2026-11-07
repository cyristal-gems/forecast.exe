"""Run expanding chronological backtests across independently trained locations."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from forecast_exe.data import NYC, geocode, synthetic, update_history
from forecast_exe.evaluation import rolling_backtest, seasonal_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", default=["New York", "London", "Singapore"])
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Single synthetic fixture, no geographic claims")
    parser.add_argument("--output", type=Path, default=Path("artifacts/backtests"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, diagnostics, provenance = [], [], []
    cities = ["SYNTHETIC fixture"] if args.demo else args.cities
    for city in cities:
        if args.demo:
            df, meta = synthetic()
        else:
            locations = [NYC] if city == "New York" else geocode(city)
            if not locations:
                parser.error(f"No location found: {city}")
            location = locations[0]
            end = datetime.now(ZoneInfo(location["timezone"])).date() - timedelta(days=1)
            df, meta = update_history(location, end - timedelta(days=args.years * 365), end)
        print(f"Evaluating {city} independently…", flush=True)
        result = rolling_backtest(df, meta, seed=args.seed, folds=args.folds)
        metrics.append(result["metrics"].assign(city=city))
        diagnostic = seasonal_diagnostics(result["predictions"]).assign(city=city)
        diagnostics.append(diagnostic)
        provenance.append(dict(city=city, source=meta, folds=result["boundaries"], synthetic=args.demo))
    pd.concat(metrics).to_csv(args.output / "metrics.csv", index=False)
    pd.concat(diagnostics).to_csv(args.output / "seasonal.csv", index=False)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Results saved to {args.output}. Check dates and sample counts before comparing cities.")


if __name__ == "__main__":
    main()
