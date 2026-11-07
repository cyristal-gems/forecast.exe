"""Local-calendar daily ingestion, validation and transparent synthetic examples."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

NYC = dict(name="New York City, NY, USA", latitude=40.7128, longitude=-74.0060, timezone="America/New_York")
FIELDS = {"temperature_2m_max": "high", "temperature_2m_min": "low", "precipitation_sum": "rain"}


def request(url, params, timeout=30, attempts=3):
    """Make a bounded request with retry for transport/server/throttling failures."""
    for attempt in range(attempts):
        try:
            response = httpx.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("error"):
                raise ValueError("Provider returned an invalid response")
            return payload
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Request failed; retry later")


def geocode(query):
    """Return explicit location choices, including timezones."""
    rows = request(
        "https://geocoding-api.open-meteo.com/v1/search", {"name": query, "count": 8, "language": "en"}
    ).get("results", [])
    return [
        dict(
            name=f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}",
            latitude=r["latitude"],
            longitude=r["longitude"],
            timezone=r["timezone"],
        )
        for r in rows
        if all(k in r for k in ("name", "latitude", "longitude", "timezone"))
    ]


def validate(df):
    """Reject missing days/values, duplicate dates and implausible target values."""
    if not len(df) or not {"high", "low", "rain"}.issubset(df):
        raise ValueError("Missing required daily variables")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    if not df.index.equals(pd.date_range(df.index.min(), df.index.max(), name=df.index.name)):
        raise ValueError("Missing calendar days; repair the source before training")
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Missing or non-finite observations")
    if ((df.low > df.high) | (df.low < -100) | (df.high > 65) | (df.rain < 0) | (df.rain > 2000)).any():
        raise ValueError("Physically implausible daily values")
    return df


def history(location, start, end, root=Path("data"), refresh=False):
    """Cache normalized and raw payloads as Parquet plus request provenance."""
    params = dict(
        latitude=location["latitude"],
        longitude=location["longitude"],
        timezone=location["timezone"],
        start_date=str(start),
        end_date=str(end),
        daily=",".join(FIELDS),
        temperature_unit="celsius",
        precipitation_unit="mm",
    )
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.parquet"
    if path.exists() and not refresh:
        return validate(pd.read_parquet(path)), json.loads(path.with_suffix(".json").read_text())
    payload = request("https://archive-api.open-meteo.com/v1/archive", params)
    daily = payload.get("daily", {})
    if not all(k in daily for k in [*FIELDS, "time"]):
        raise ValueError("Historical API omitted required daily variables")
    units = payload.get("daily_units", {})
    if any(units.get(k) != ("mm" if k == "precipitation_sum" else "°C") for k in FIELDS):
        raise ValueError("Unexpected provider units")
    df = pd.DataFrame(
        {v: daily[k] for k, v in FIELDS.items()}, index=pd.DatetimeIndex(daily["time"], name="date")
    )
    validate(df)
    meta = dict(
        location=location,
        request=params,
        units=units,
        source="Open-Meteo reanalysis",
        retrieved_at=datetime.now(UTC).isoformat(),
        synthetic=False,
    )
    pd.DataFrame(daily).to_parquet(root / f"{key}-raw.parquet")
    df.to_parquet(path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return df, meta


def synthetic():
    """Deterministic synthetic fixture; never represents actual weather."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2019-01-01", periods=1826, name="date")
    season = 13 + 13 * np.sin(2 * np.pi * (dates.strftime("%j").astype(int).to_numpy() - 105) / 365.25)
    noise = np.zeros(len(dates))
    for i in range(1, len(dates)):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 2)
    df = pd.DataFrame(
        dict(
            high=season + noise + 5,
            low=season + noise - 4,
            rain=rng.exponential(5, len(dates)) * (rng.random(len(dates)) < 0.32),
        ),
        index=dates,
    )
    return df, dict(
        source="SYNTHETIC demonstration",
        synthetic=True,
        location=NYC,
        units={"temperature": "°C", "precipitation": "mm"},
    )


def update_history(location, start, end, root=Path("data"), refresh=False):
    """Reuse overlapping local data; request only missing ranges and seven recent days."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end < start:
        raise ValueError("Choose an end date on or after the start")
    key = hashlib.sha256(json.dumps(location, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    master = root / f"location-{key}.parquet"
    metadata = master.with_suffix(".json")
    current = (
        pd.read_parquet(master)
        if master.exists()
        else pd.DataFrame(columns=["high", "low", "rain"], index=pd.DatetimeIndex([], name="date"))
    )
    previous = json.loads(metadata.read_text()) if metadata.exists() else None
    wanted = pd.date_range(start, end, name="date")
    missing = wanted.difference(current.index)
    if refresh:
        missing = missing.union(wanted[-7:])
    ranges = []
    for day in missing:
        if not ranges or day != ranges[-1][1] + pd.Timedelta(days=1):
            ranges.append([day, day])
        else:
            ranges[-1][1] = day
    fetched, requests = [], list(previous.get("segments", [])) if previous else []
    for first, last in ranges:
        frame, meta = history(location, first.date(), last.date(), root / "requests", refresh=refresh)
        fetched.append(frame)
        requests.append(meta)
    if fetched:
        parts = [current] if len(current) else []
        combined = pd.concat(parts + fetched)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = current
    result = combined.reindex(wanted)
    validate(result)
    meta = dict(
        location=location,
        synthetic=False,
        source="Open-Meteo reanalysis",
        units={"high": "°C", "low": "°C", "rain": "mm"},
        date_range=[str(start.date()), str(end.date())],
        retrieved_at=datetime.now(UTC).isoformat() if fetched else previous["retrieved_at"],
        segments=requests,
        update_policy="missing date ranges + optional seven-day refresh",
    )
    combined.to_parquet(master)
    metadata.write_text(json.dumps(meta, indent=2))
    return result, meta


def cached_history(location, start, end, root=Path("data")):
    """Restore an existing complete date window without making any network request."""
    key = hashlib.sha256(json.dumps(location, sort_keys=True).encode()).hexdigest()[:20]
    path = Path(root) / f"location-{key}.parquet"
    if not path.exists() or not path.with_suffix(".json").exists():
        return None
    try:
        index = pd.date_range(start, end, name="date")
        df = pd.read_parquet(path).reindex(index)
        validate(df)
        meta = json.loads(path.with_suffix(".json").read_text())
        meta["date_range"] = [str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())]
        return df, meta
    except (ValueError, KeyError, OSError):
        return None
