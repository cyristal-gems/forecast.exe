"""Weather-first presentation of clearly attributed external current estimates."""

from html import escape
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import streamlit as st

from forecast_exe.weather import current_weather, condition
from forecast_exe.themes import apply_background


@st.cache_data(ttl=600, show_spinner=False)
def weather_snapshot(location):
    return current_weather(location)


def cards(frame, fahrenheit, external=False):
    unit = "°F" if fahrenheit else "°C"
    html = []
    for row in frame.itertuples():
        high, low = (row.high * 1.8 + 32, row.low * 1.8 + 32) if fahrenheit else (row.high, row.low)
        note = "Max hourly rain chance" if external else "Daily rain ≥ 0.1 mm"
        html.append(
            f'<article class="day"><b>{row.date:%a}</b><div class="date">{row.date:%b %d}</div>'
            f'<div class="temp">{high:.0f}{unit}</div><div class="low">Low {low:.0f}{unit}</div>'
            f'<div class="rain">Rain {row.rain:.0%}<small>{note}</small></div>'
            f'<div class="rain-bar"><i style="width:{row.rain * 100:.1f}%"></i></div></article>'
        )
    st.markdown('<div class="cards">' + "".join(html) + "</div>", unsafe_allow_html=True)


@st.fragment(run_every=600)
def external_weather(location, fahrenheit, compact=False):
    """Render provider output even when no historical data or local model exists."""
    if st.button("Refresh weather", key="refresh_external_weather"):
        weather_snapshot.clear(location)
    try:
        with st.spinner("Getting current weather…"):
            snapshot = weather_snapshot(location)
    except Exception as exc:
        st.warning(
            f"External weather is unavailable: {exc}. Retry, search for another city, "
            "or explicitly choose Synthetic demo. Local model training is available in Model Studio."
        )
        return None
    apply_background(location, snapshot=snapshot)
    now, frame = snapshot["current"], snapshot["daily"]
    if (datetime.now(UTC) - datetime.fromisoformat(snapshot["valid_at_utc"])).total_seconds() > 3 * 3600:
        st.warning(
            "The provider’s current estimate is over three hours old. Check its valid time before using it."
        )

    def convert(value):
        return value * 1.8 + 32 if fahrenheit else value

    unit = "°F" if fahrenheit else "°C"
    description, icon = condition(now["weather_code"], bool(now["is_day"]))
    local_clock = datetime.now(ZoneInfo(location["timezone"])).strftime("%A, %B %d · %I:%M %p")
    st.markdown(
        f'<section class="weather-now"><div class="eyebrow">External weather / Open-Meteo</div>'
        f'<h2>{escape(location["name"])}</h2><div class="weather-clock">{local_clock}</div>'
        f'<div class="weather-reading"><span class="weather-icon" aria-hidden="true">{icon}</span>'
        f"<strong>{convert(now['temperature_2m']):.0f}<small>{unit}</small></strong></div>"
        f'<div class="weather-condition">{description}</div>'
        f'<div class="weather-details"><span>Feels like {convert(now["apparent_temperature"]):.0f}{unit}</span>'
        f"<span>Humidity {now['relative_humidity_2m']:.0f}%</span>"
        f"<span>Wind {now['wind_speed_10m']:.0f} km/h</span></div></section>",
        unsafe_allow_html=True,
    )
    updated = datetime.fromisoformat(now["time"]).strftime("%I:%M %p")
    st.markdown(
        f'<p class="weather-timing">External model estimate · Updated {updated} · '
        f'{escape(snapshot["timezone"])} · Refreshes every 10 minutes while open</p>',
        unsafe_allow_html=True,
    )
    if not compact:
        st.markdown("### Seven-day forecast")
        st.caption("EXTERNAL OPERATIONAL FORECAST · Open-Meteo · Independent of local training")
        cards(frame, fahrenheit, external=True)
        st.caption(
            "Rain percentage is the maximum hourly precipitation probability during each day; "
            "it has a different definition from the local model’s daily rain-event probability."
        )
        st.download_button(
            "Export external forecast", frame.to_csv(index=False), "external-weather.csv", "text/csv"
        )
    return snapshot
