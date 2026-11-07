"""Download historical data independently of the UI."""

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from forecast_exe.data import NYC, geocode, update_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="New York")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--refresh", action="store_true", help="Also refresh the latest seven days")
    args = parser.parse_args()
    locations = [NYC] if args.city == "New York" else geocode(args.city)
    if not locations:
        parser.error("No location found")
    loc = locations[0]
    end = datetime.now(ZoneInfo(loc["timezone"])).date() - timedelta(days=1)
    df, meta = update_history(loc, end - timedelta(days=365 * args.years), end, refresh=args.refresh)
    print(f"{loc['name']}: {len(df)} complete days through {df.index[-1].date()}")
    print(meta)


if __name__ == "__main__":
    main()
