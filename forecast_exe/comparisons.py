"""City comparison uses external forecasts only, with explicit local-day alignment."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from html import escape

from forecast_exe.data import NYC
from forecast_exe.locations import validate_location
from forecast_exe.weather import condition
from forecast_exe.weather_ui import weather_snapshot
from forecast_exe.themes import apply_background

LONDON = dict(
    name="London, England, United Kingdom", latitude=51.50853, longitude=-0.12574, timezone="Europe/London"
)
SINGAPORE = dict(
    name="Singapore, Singapore", latitude=1.28967, longitude=103.85007, timezone="Asia/Singapore"
)


def city_key(city):
    return f"{city['latitude']:.4f},{city['longitude']:.4f}:{city['timezone']}"


def city_library(state, active):
    cities = [NYC, LONDON, SINGAPORE, active, *state.get("compare_locations", [])]
    result = {city_key(city): validate_location(city) for city in cities}
    state["compare_locations"] = list(result.values())
    return result


def add_to_comparison(state, city):
    city = validate_location(city)
    options = city_library(state, state.get("active_location", NYC))
    options[city_key(city)] = city
    state["compare_locations"] = list(options.values())
    selected = list(state.get("compare_city_ids", [city_key(NYC), city_key(LONDON)]))
    if city_key(city) not in selected and len(selected) < 4:
        selected.append(city_key(city))
    state["compare_city_ids"] = selected
    state["workspace"] = "Compare Cities"


def comparison_frame(snapshots):
    """Keep timezone and source on every daily row; never imply UTC alignment."""
    rows = []
    for city, snapshot in snapshots:
        frame = snapshot["daily"].copy()
        frame["city"] = city["name"]
        frame["location_id"] = city_key(city)
        frame["latitude"] = city["latitude"]
        frame["longitude"] = city["longitude"]
        frame["timezone"] = city["timezone"]
        frame["source"] = "Open-Meteo external operational forecast"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def render_comparison(active, fahrenheit, demo):
    if demo:
        st.info(
            "Compare Cities uses real external weather. Turn off Synthetic demo in the gear menu to continue."
        )
        return
    options = city_library(st.session_state, active)
    defaults = list(dict.fromkeys([city_key(active), city_key(LONDON)]))
    if len(defaults) < 2:
        defaults.append(city_key(NYC))
    selected = st.multiselect(
        "Cities to compare",
        list(options),
        default=defaults,
        format_func=lambda value: options[value]["name"],
        max_selections=4,
        key="compare_city_ids",
    )
    st.markdown(
        "Choose **2–4 cities**. To add another, open **⚙**, search for the city, "
        "and click **Add to comparison**."
    )
    if len(selected) < 2:
        st.info("Choose at least two cities for a comparison.")
        return
    if st.button("Refresh city comparison"):
        for key in selected:
            weather_snapshot.clear(options[key])
    snapshots = []
    for key in selected:
        city = options[key]
        try:
            with st.spinner("Getting weather for " + city["name"] + "…"):
                snapshots.append((city, weather_snapshot(city)))
        except Exception as exc:
            st.warning(f"{city['name']}: weather unavailable ({exc}). Other cities remain visible.")
    if not snapshots:
        return
    unit = "°F" if fahrenheit else "°C"
    for column, (city, snapshot) in zip(st.columns(len(snapshots)), snapshots):
        now = snapshot["current"]
        value = now["temperature_2m"] * 1.8 + 32 if fahrenheit else now["temperature_2m"]
        description, symbol = condition(now["weather_code"], bool(now["is_day"]))
        with column:
            st.markdown(
                f'<section class="compare-card"><div class="eyebrow">External current estimate</div>'
                f'<h3>{escape(city["name"])}</h3><div class="compare-temperature">{value:.0f}{unit}</div>'
                f'<p>{symbol} {description}</p><div class="date">Humidity {now["relative_humidity_2m"]:.0f}% · '
                f"Wind {now['wind_speed_10m']:.0f} km/h</div></section>",
                unsafe_allow_html=True,
            )
            st.caption(f"Valid {now['time']} · {city['timezone']}")
    for city, snapshot in snapshots:
        if city_key(city) == city_key(active):
            apply_background(active, snapshot=snapshot)
            break
    frame = comparison_frame(snapshots)
    temperature_tab, rain_tab, table_tab = st.tabs(["Temperature outlook", "Rain chance", "Daily table"])
    colors = ["#d6e9f6", "#a6d5c7", "#efc7a0", "#b5b1ed"]
    short_names = [city["name"].split(",")[0] for city, _ in snapshots]
    labels = {
        city_key(city): city["name"].split(",")[0]
        if short_names.count(city["name"].split(",")[0]) == 1
        else city["name"].split(",")[0] + f" ({city['latitude']:.2f}, {city['longitude']:.2f})"
        for city, _ in snapshots
    }
    with temperature_tab:
        fig = go.Figure()
        for index, (city, _) in enumerate(snapshots):
            data = frame[frame.location_id == city_key(city)]
            for target, dash in [("high", "solid"), ("low", "dash")]:
                values = data[target] * 1.8 + 32 if fahrenheit else data[target]
                fig.add_trace(
                    go.Scatter(
                        x=data.date,
                        y=values,
                        name=labels[city_key(city)] + " " + target,
                        hovertemplate=escape(city["name"])
                        + "<br>%{x|%b %d}: %{y:.1f}<extra>"
                        + target
                        + "</extra>",
                        line=dict(color=colors[index], dash=dash),
                        mode="lines+markers",
                    )
                )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title=unit,
            xaxis_title="Local calendar day",
            height=440,
            margin=dict(t=25, l=40, r=25, b=95),
            showlegend=False,
            xaxis=dict(title=dict(text="Local calendar day", standoff=25), ticklabelstandoff=8),
        )
        st.plotly_chart(fig, width="stretch")
        render_key(
            [(labels[city_key(city)] + " high", colors[i], False) for i, (city, _) in enumerate(snapshots)]
            + [(labels[city_key(city)] + " low", colors[i], True) for i, (city, _) in enumerate(snapshots)]
        )
    with rain_tab:
        rain_figure = go.Figure()
        for index, (city, _) in enumerate(snapshots):
            data = frame[frame.location_id == city_key(city)]
            rain_figure.add_trace(
                go.Bar(
                    x=data.date, y=data.rain * 100, name=labels[city_key(city)], marker_color=colors[index]
                )
            )
        rain_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Maximum hourly rain chance (%)",
            yaxis_range=[0, 100],
            xaxis_title="Local calendar day",
            barmode="group",
            height=420,
            margin=dict(t=25, l=40, r=25, b=95),
            showlegend=False,
            xaxis=dict(title=dict(text="Local calendar day", standoff=25), ticklabelstandoff=8),
        )
        st.plotly_chart(rain_figure, width="stretch")
        render_key([(labels[city_key(city)], colors[i], False) for i, (city, _) in enumerate(snapshots)])
    with table_tab:
        display = frame.copy()
        if fahrenheit:
            display[["high", "low"]] = display[["high", "low"]] * 1.8 + 32
            display["temperature_unit"] = "°F"
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption(f"Temperatures shown in {unit}; rain is a probability between 0 and 1.")
    st.caption(
        "External forecasts, not local ML or accuracy scores. Dates are local to each city, not synchronized "
        "UTC intervals. Current valid times are shown individually. Rain is the daily maximum of hourly probabilities."
    )
    st.download_button("Export comparison", frame.to_csv(index=False), "city-comparison.csv", "text/csv")
    st.caption("CSV temperatures remain in °C; every row retains its city, timezone, and external source.")


def render_key(items):
    html = '<section class="chart-key" aria-label="Chart key"><div class="eyebrow">Chart key</div><div class="key-items">'
    for label, color, dashed in items:
        html += (
            '<span class="key-item"><i style="border-top:2px '
            + ("dashed" if dashed else "solid")
            + " "
            + color
            + '"></i>'
            + escape(label)
            + "</span>"
        )
    st.markdown(html + "</div></section>", unsafe_allow_html=True)
