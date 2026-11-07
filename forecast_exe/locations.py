"""Offline nearby-city resolution and validation of browser preferences."""

import math
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def coordinates(latitude, longitude):
    """Reject malformed browser input before any lookup or weather request."""
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        raise ValueError("Invalid coordinates")
    try:
        lat, lon = float(latitude), float(longitude)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid coordinates") from exc
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Coordinates are outside the supported range")
    return lat, lon


def validate_location(value):
    """Saved preferences are untrusted; preserve only the application's city schema."""
    if not isinstance(value, dict):
        raise ValueError("Saved city is invalid; search for a city instead")
    lat, lon = coordinates(value.get("latitude"), value.get("longitude"))
    name, timezone = value.get("name"), value.get("timezone")
    if not isinstance(name, str) or not name.strip() or len(name) > 180:
        raise ValueError("Invalid city name")
    if not isinstance(timezone, str) or len(timezone) > 80:
        raise ValueError("Invalid city timezone")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("Unsupported city timezone") from exc
    return dict(name=name.strip(), latitude=lat, longitude=lon, timezone=timezone)


@lru_cache(maxsize=1)
def city_index():
    """Load bundled GeoNames cities once; no third-party reverse-location request."""
    import reverse_geocoder

    return reverse_geocoder.RGeocoder(mode=1, verbose=False)


def nearby_city(latitude, longitude):
    """Return a nearby populated place's center, region, country, and timezone."""
    import pycountry
    from tzfpy import get_tz

    lat, lon = coordinates(latitude, longitude)
    place = city_index().query([(lat, lon)])[0]
    center_lat, center_lon = coordinates(place["lat"], place["lon"])
    # Great-circle distance guards against presenting a remote city as local.
    a, b = math.radians(lat), math.radians(center_lat)
    dlat, dlon = b - a, math.radians(center_lon - lon)
    distance = (
        6371
        * 2
        * math.asin(
            min(1, math.sqrt(math.sin(dlat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dlon / 2) ** 2))
        )
    )
    if distance > 100:
        raise ValueError("No nearby city within 100 km. Please choose a city with manual search.")
    country = pycountry.countries.get(alpha_2=place["cc"])
    country_name = country.name if country else place["cc"]
    parts = [place["name"], place.get("admin1", ""), country_name]
    name = ", ".join(dict.fromkeys(part for part in parts if part))
    return validate_location(
        dict(name=name, latitude=center_lat, longitude=center_lon, timezone=get_tz(center_lon, center_lat))
    )
