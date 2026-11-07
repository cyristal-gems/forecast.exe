"""Explicit reproducible model training and inference CLI."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from forecast_exe.data import NYC, geocode, synthetic, update_history
from forecast_exe.model import forecast, train
from forecast_exe.experiments import register


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="New York")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.horizons != list(range(1, 8)):
        parser.error("This release requires all seven horizons: 1 2 3 4 5 6 7")
    if args.demo:
        df, meta = synthetic()
        loc = NYC
    else:
        locations = [NYC] if args.city == "New York" else geocode(args.city)
        if not locations:
            parser.error("No location found")
        loc = locations[0]
        end = datetime.now(ZoneInfo(loc["timezone"])).date() - timedelta(days=1)
        df, meta = update_history(loc, end - timedelta(days=365 * args.years), end)
    bundle = train(df, meta, args.seed)
    run_id = register(bundle, args.output)
    print(f"Saved experiment: {run_id}")
    print(bundle["metrics"].to_string(index=False))
    try:
        result = forecast(bundle, df, loc, demo=args.demo)
        result.to_csv(args.output / f"{run_id}.csv", index=False)
        print(result.to_string(index=False))
    except ValueError as exc:
        print(f"Training and evaluation complete; inference unavailable: {exc}")


if __name__ == "__main__":
    main()
