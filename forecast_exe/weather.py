"""Validated external current estimates and operational daily weather, never ML labels."""

from datetime import UTC, datetime, timezone, timedelta

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from forecast_exe.data import request


def current_weather(location):
    """Fetch explicit metric units and retain current/valid/retrieval timestamps."""
    payload = request(
        "https://api.open-meteo.com/v1/forecast",
        dict(
            latitude=location["latitude"],
            longitude=location["longitude"],
            timezone=location["timezone"],
            current="temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,is_day",
            daily="temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            forecast_days=7,
            temperature_unit="celsius",
            wind_speed_unit="kmh",
            precipitation_unit="mm",
        ),
        timeout=10,
        attempts=2,
    )
    try:
        daily, current = payload["daily"], payload["current"]
        frame = pd.DataFrame(
            dict(
                date=pd.to_datetime(daily["time"]),
                high=daily["temperature_2m_max"],
                low=daily["temperature_2m_min"],
                rain=np.asarray(daily["precipitation_probability_max"], dtype=float) / 100,
            )
        )
        frame[["high", "low", "rain"]] = frame[["high", "low", "rain"]].astype(float)
        numeric = frame[["high", "low", "rain"]].to_numpy(dtype=float)
        if (
            len(frame) != 7
            or frame.date.duplicated().any()
            or not frame.date.is_monotonic_increasing
            or not frame.date.diff().dropna().eq(pd.Timedelta(days=1)).all()
        ):
            raise ValueError("Expected seven distinct daily forecast dates")
        if (
            not np.isfinite(numeric).all()
            or not frame.rain.between(0, 1).all()
            or (frame.low > frame.high).any()
        ):
            raise ValueError("Incomplete daily weather; please retry later")
        expected = {
            "temperature_2m_max": "°C",
            "temperature_2m_min": "°C",
            "precipitation_probability_max": "%",
        }
        if any(payload["daily_units"].get(key) != value for key, value in expected.items()):
            raise ValueError("Unexpected daily-weather units")
        expected_current = {
            "temperature_2m": "°C",
            "apparent_temperature": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "km/h",
        }
        if any(payload["current_units"].get(key) != value for key, value in expected_current.items()):
            raise ValueError("Unexpected current-weather units")
        if payload.get("timezone") != location["timezone"]:
            raise ValueError("Provider returned a different timezone; retry later")
        fields = [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "wind_speed_10m",
            "weather_code",
            "is_day",
        ]
        if (
            any(current.get(k) is None for k in fields)
            or not np.isfinite([float(current[k]) for k in fields]).all()
        ):
            raise ValueError("Current weather is incomplete")
        current = dict(current, **{k: float(current[k]) for k in fields})
        pd.Timestamp(current["time"])
        if (
            not 0 <= current["relative_humidity_2m"] <= 100
            or current["wind_speed_10m"] < 0
            or current["is_day"] not in [0, 1]
            or not current["weather_code"].is_integer()
        ):
            raise ValueError("Invalid current-weather readings")
        offset = int(payload["utc_offset_seconds"])
        if abs(offset) > 14 * 3600:
            raise ValueError("Invalid provider UTC offset")
        valid_time = datetime.fromisoformat(current["time"])
        if valid_time.tzinfo is None:
            allowed_offsets = {
                int(
                    valid_time.replace(tzinfo=ZoneInfo(location["timezone"]), fold=fold)
                    .utcoffset()
                    .total_seconds()
                )
                for fold in [0, 1]
            }
            if offset not in allowed_offsets:
                raise ValueError("Provider time offset does not match the selected city")
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=timezone(timedelta(seconds=offset)))
        fetched = datetime.now(UTC).isoformat()
        frame = frame.assign(
            city=location["name"],
            timezone=location["timezone"],
            source="Open-Meteo external operational forecast",
            temperature_unit="°C",
            rain_definition="Daily maximum hourly precipitation probability",
            rain_probability_unit="fraction (0–1)",
            fetched_at=fetched,
        )
        return dict(
            valid_at_utc=valid_time.astimezone(UTC).isoformat(),
            daily=frame,
            current=current,
            timezone=payload.get("timezone", location["timezone"]),
            fetched_at=fetched,
            source="Open-Meteo external operational weather",
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("External weather is incomplete; retry later") from exc


def condition(code, is_day=True):
    """WMO descriptions are used only for provider-supplied current weather codes."""
    code = int(code)
    if code == 0:
        return ("Clear sky", "☀" if is_day else "☾")
    if code in (1, 2):
        return ("Mainly clear" if code == 1 else "Partly cloudy", "◐")
    if code == 3:
        return ("Overcast", "☁")
    if code in (45, 48):
        return ("Fog", "≋")
    if code in (51, 53, 55, 56, 57):
        return ("Drizzle", "☂")
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return ("Rain", "☂")
    if code in (71, 73, 75, 77, 85, 86):
        return ("Snow", "❄")
    if code in (95, 96, 99):
        return ("Thunderstorm", "ϟ")
    return ("Conditions unavailable", "◌")
