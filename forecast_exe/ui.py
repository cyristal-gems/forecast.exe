"""Guided weather studio with immutable experiments and asynchronous training."""

from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast_exe.data import cached_history, update_history, request, synthetic
from forecast_exe.evaluation import seasonal_diagnostics
from forecast_exe.experiments import Jobs, compatible_runs, list_runs, get_run, experiment_table
from forecast_exe.model import VERSION, fingerprint, forecast
from forecast_exe import __version__ as APP_VERSION
from forecast_exe.location_ui import location_button, search_controls
from forecast_exe.weather_ui import cards, external_weather
from forecast_exe.comparisons import render_comparison
from forecast_exe.themes import apply_background
from forecast_exe.branding import contact_dock


@st.cache_resource
def job_manager():
    return Jobs()


@st.cache_data
def demo_data():
    return synthetic()


def temperature_chart(frame, fahrenheit=False, intervals=False):
    def convert(x):
        return x * 1.8 + 32 if fahrenheit else x

    fig = go.Figure()
    for target, color, fill in [
        ("high", "#ffbd79", "rgba(255,189,121,.12)"),
        ("low", "#6ebfff", "rgba(110,191,255,.12)"),
    ]:
        if intervals:
            fig.add_trace(
                go.Scatter(
                    x=frame.date,
                    y=convert(frame[f"{target}_lower"]),
                    line=dict(width=0),
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=frame.date,
                    y=convert(frame[f"{target}_upper"]),
                    fill="tonexty",
                    fillcolor=fill,
                    line=dict(width=0),
                    mode="lines",
                    name=f"{target.title()} approximate 80% interval",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=frame.date,
                y=convert(frame[target]),
                name=target.title(),
                mode="lines+markers",
                line=dict(color=color, width=3),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a9bed5"),
        height=540,
        margin=dict(l=15, r=15, t=20, b=180),
        yaxis_title="°F" if fahrenheit else "°C",
        legend=dict(orientation="h", x=0, y=-0.22, yanchor="top", font=dict(size=11)),
    )
    return fig


def hero(page, demo, compact=False):
    titles = {
        "Forecast": (
            "A clearer view of<br>what comes next.",
            "Seven days. Your own models. Every uncertainty in view.",
        ),
        "Historical Weather": (
            "Every forecast starts<br>with a little history.",
            "Explore the daily patterns that teach your models.",
        ),
        "Model Lab": (
            "Trust is built<br>on evidence.",
            "Explore held-out accuracy, uncertainty, and baseline comparisons.",
        ),
        "Experiments": (
            "Progress, measured.<br>Every run, remembered.",
            "Compare saved experiments without losing the context behind the scores.",
        ),
        "Methodology": (
            "Open by design.<br>Honest by default.",
            "Understand the data, the models, and their limits.",
        ),
    }
    title, subtitle = titles[page]
    st.markdown(
        f'<section class="hero {"compact" if compact else ""}"><div class="eyebrow">Independent weather intelligence / {APP_VERSION}</div>'
        f'<h2>{title}</h2><p>{subtitle}</p><span class="tag">'
        f"{'SYNTHETIC DEMONSTRATION' if demo else 'LOCAL MODELS · REAL HISTORICAL DATA'}"
        "</span></section>",
        unsafe_allow_html=True,
    )


def steps(loaded, trained):
    labels = [
        ("01", "Choose a city", "Start with a place", True),
        ("02", "Load weather", "Validated daily history", loaded),
        ("03", "Train your models", "Local & reproducible", trained),
        ("04", "Explore your forecast", "Seven days with context", trained),
    ]
    html = '<div class="steps">'
    for number, title, subtitle, done in labels:
        html += f'<div class="step {"done" if done else "active"}"><b>{number} / {title}</b>{subtitle}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


@st.fragment(run_every=1)
def job_status(token):
    manager = job_manager()
    state = manager.snapshot(token)
    if state["status"] in ["queued", "running"]:
        st.progress(state["progress"], text="Training locally… You can explore other pages while this runs.")
        if st.button("Cancel training", key="cancel-" + token):
            manager.cancel(token)
            st.info("Cancellation requested; the current horizon finishes before stopping.")
    else:
        if state["status"] == "failed":
            st.error("Training failed: " + state["error"])
        elif state["status"] == "cancelled":
            st.info("Training cancelled. No partial model was saved.")
        else:
            st.success(
                "Experiment complete and saved." if state.get("run") else "Rolling evaluation complete."
            )
        if token not in st.session_state.get("finished_jobs", []):
            st.session_state["finished_jobs"] = st.session_state.get("finished_jobs", []) + [token]
            if state.get("run"):
                st.session_state["latest_run"] = state["run"]
            st.rerun()


def open_model_lab():
    st.session_state["workspace"] = "Model Studio"


def open_local_forecast():
    st.session_state["workspace"] = "Weather"
    st.session_state["forecast_source"] = "Local ML"


def local_model_empty(location, fahrenheit):
    st.subheader("Your local forecast is not ready yet")
    st.info(
        "Open Model Studio to load history and train models for this city. "
        "The separate external weather view is available immediately."
    )
    st.button("Prepare my local forecast", on_click=open_model_lab, type="primary")
    token = st.session_state.get("training_job")
    if token and (
        token not in st.session_state.get("finished_jobs", [])
        or job_manager().snapshot(token)["status"] in ["queued", "running"]
    ):
        job_status(token)
    with st.expander("Separate external current weather", expanded=True):
        external_weather(location, fahrenheit, compact=True)


NAVIGATION = {
    "Weather": "Forecast",
    "Weather History": "Historical Weather",
    "Model Studio": "Model Lab",
    "Compare Cities": "Compare Cities",
    "How It Works": "Methodology",
}

TAB_SUMMARIES = {
    "Forecast": "See current weather estimates and the seven-day outlook for your chosen city. "
    "Use the location arrow to find a nearby city, or choose a place from the gear menu. "
    "The external forecast is available before you train a model. "
    "Switch to Local ML to see predictions learned from that city’s history. "
    "Change temperature units with the header pill to suit your preference.",
    "Historical Weather": "Explore the daily weather records behind your local models. "
    "Load a city’s history and choose the date range you want to inspect. "
    "Temperature and precipitation charts reveal patterns across days and seasons. "
    "Dataset details show completeness, units, and where the reference values came from. "
    "Use this view to understand the past before interpreting a new prediction.",
    "Model Lab": "Train weather models specifically for the city you selected. "
    "Choose a history window and start a reproducible training run. "
    "Compare held-out results with persistence and seasonal baselines. "
    "Inspect rain calibration and temperature uncertainty across forecast horizons and seasons. "
    "Open Saved Runs within this tab to revisit and compare completed experiments.",
    "Compare Cities": "Compare current estimates and upcoming weather for up to four cities. "
    "Add cities through the gear menu and choose the places you want to compare. "
    "Read temperatures, rain chances, humidity, and wind side by side. "
    "All comparisons in this view use external Open-Meteo forecasts. "
    "Each date follows the city’s local timezone, so the view compares local calendar days rather than identical instants.",
    "Methodology": "Understand how the app turns historical weather into local predictions. "
    "Learn how external forecasts, reanalysis references, and local ML outputs differ. "
    "Review the chronological evaluation, baseline definitions, and uncertainty methods. "
    "See why data freshness and location-specific training matter. "
    "Use this project for exploration and learning rather than safety-critical weather decisions.",
}


def tab_summary(page):
    st.markdown(
        '<section class="tab-summary"><p>' + escape(TAB_SUMMARIES[page]) + "</p></section>",
        unsafe_allow_html=True,
    )


def main(asset_root=None):
    asset_root = Path(asset_root or Path(__file__).resolve().parents[1] / "static")
    st.set_page_config(
        page_title="Forecast.exe", page_icon=(asset_root / "forecast-mark.svg").read_text(), layout="wide"
    )
    st.markdown(
        "<style>" + Path(__file__).with_name("style.css").read_text() + "</style>", unsafe_allow_html=True
    )
    st.title("Forecast.exe")
    with st.container(key="app-header"):
        with st.container(key="header-controls"):
            brand, arrow, temperature, gear = st.columns(
                [6, 1, 1.2, 1], gap="small", vertical_alignment="center", wrap=False
            )
            with brand:
                st.markdown(
                    '<a class="site-brand" href="./" target="_self" aria-label="Reload Forecast.exe"><img src="app/static/forecast-logo.svg" alt="Forecast.exe" width="230"></a>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="brand-tagline">Decoding the Skies, One Prediction at a Time.</p>',
                    unsafe_allow_html=True,
                )
            with arrow:
                location = location_button()
            with temperature:
                with st.popover(
                    st.session_state.get("temperature_unit", "°C"),
                    key="temperature_menu",
                    help="Temperature units",
                    width="stretch",
                ):
                    st.radio("Temperature", ["°C", "°F"], key="temperature_unit", horizontal=True)
                fahrenheit = st.session_state.get("temperature_unit", "°C") == "°F"
            with gear:
                settings = st.popover(
                    "⚙", key="settings_menu", help="City search and advanced settings", width="stretch"
                )
        with settings:
            search_controls()
            st.divider()
            st.markdown("#### App preferences")
            demo = st.toggle(
                "Synthetic demo",
                value=False,
                key="demo_mode",
                help="Generated historical weather, not real current conditions.",
            )
            st.markdown("#### Advanced model settings")
            seed = st.number_input("Training seed", value=42, min_value=0, max_value=2147483647)
            years = st.slider("History (years)", 2, 10, 5)
            start, end = None, None
            if not demo:
                today = datetime.now(ZoneInfo(location["timezone"])).date()
                end = st.date_input(
                    "Last historical day", today - timedelta(days=1), max_value=today - timedelta(days=1)
                )
                start = st.date_input(
                    "First historical day", end - timedelta(days=years * 365), max_value=end
                )
            st.caption(location["timezone"])
            st.markdown("[Weather data: Open-Meteo](https://open-meteo.com/) · CC BY 4.0")
        previous = st.session_state.get("workspace")
        reverse = {value: key for key, value in NAVIGATION.items()}
        if previous in ["Saved Runs", "Experiments"]:
            st.session_state["workspace"] = "Model Studio"
        elif previous in reverse:
            st.session_state["workspace"] = reverse[previous]
        tabs = st.tabs(list(NAVIGATION), key="workspace", on_change="rerun")
    active = next((tab for tab in tabs if tab.open), tabs[0])
    apply_background(location, demo=demo)
    with active:
        render_page(
            NAVIGATION[st.session_state.get("workspace", "Weather")],
            location,
            demo,
            fahrenheit,
            seed,
            years,
            start,
            end,
            settings,
        )
    contact_dock()


def render_page(page, location, demo, fahrenheit, seed, years, start, end, settings):
    tab_summary(page)
    if demo:
        st.caption("SYNTHETIC DEMO · Generated historical weather and evaluation; not a live forecast.")
    if st.session_state.get("location_error"):
        st.warning(st.session_state["location_error"])
    if st.session_state.get("storage_notice"):
        st.warning(st.session_state["storage_notice"])
    if page == "Forecast" and not demo:
        source = st.radio(
            "Forecast source", ["External forecast", "Local ML"], key="forecast_source", horizontal=True
        )
        if source == "External forecast":
            if not st.session_state.get("city_preferences_ready"):
                st.info(
                    "Restoring your preferred city… You can also use the location arrow or search under ⚙."
                )
                return
            external_weather(location, fahrenheit)
            has_model = bool(list_runs(location=location, demo=False))
            st.button(
                "Open my local forecast" if has_model else "Build my local forecast",
                on_click=open_local_forecast if has_model else open_model_lab,
            )
            st.caption(
                "External weather works independently. Local ML requires validated history "
                "and models trained specifically for this location."
            )
            token = st.session_state.get("training_job")
            if token and (
                token not in st.session_state.get("finished_jobs", [])
                or job_manager().snapshot(token)["status"] in ["queued", "running"]
            ):
                job_status(token)
            return
    if page == "Methodology":
        about()
        return
    if page == "Compare Cities":
        render_comparison(location, fahrenheit, demo)
        return
    if page == "Model Lab":
        studio_tabs = st.tabs(["Train & evaluate", "Saved Runs"], key="studio_section", on_change="rerun")
        if studio_tabs[1].open:
            with studio_tabs[1]:
                experiments_page(location, demo)
            return
    if demo:
        df, meta = demo_data()
    else:
        key = str([location, start, end])
        stored = st.session_state.get("dataset")
        if not stored or stored[0] != key:
            recovered = cached_history(location, start, end)
            if recovered:
                st.session_state["dataset"] = (key, *recovered)
                stored = st.session_state["dataset"]
        loaded = bool(stored and stored[0] == key)
        if not loaded and page == "Forecast":
            local_model_empty(location, fahrenheit)
            return
        if not loaded:
            steps(False, False)
            st.subheader("Your place. Your forecast.")
            st.write(
                f"Load {years} years of complete daily weather for {location['name']}. "
                "Existing history is reused; only missing date ranges are downloaded."
            )
        load_clicked = settings.button(
            "Load weather" if not loaded else "Refresh recent days", type="primary"
        )
        if not loaded:
            load_clicked = st.button("Load weather & continue", type="primary") or load_clicked
        if load_clicked:
            try:
                with st.spinner("Loading and validating daily history…"):
                    df, meta = update_history(location, start, end, refresh=loaded)
                    st.session_state["dataset"] = (key, df, meta)
                st.rerun()
            except Exception as exc:
                st.error(
                    f"Weather unavailable: {exc}. Your valid local history is retained. "
                    "Try again later or explicitly choose synthetic demo."
                )
        stored = st.session_state.get("dataset")
        if not stored or stored[0] != key:
            return
        _, df, meta = stored
    compatible = compatible_runs(df, location, demo, seed)
    bundle = None
    if compatible:
        ids = [m["run_id"] for m in compatible]
        preferred = st.session_state.get("latest_run")
        index = ids.index(preferred) if preferred in ids else 0
        selected = settings.selectbox("Active experiment", ids, index=index)
        try:
            bundle = get_run(selected)
        except (ValueError, OSError) as exc:
            st.error(f"{exc}. Train a new experiment to recover.")
    if bundle is None:
        steps(True, False)
    st.subheader(location["name"])
    ready_forecast = page == "Forecast" and bundle is not None
    if not ready_forecast:
        a, b, c = st.columns(3)
        a.metric("Complete history", f"{len(df):,} days")
        b.metric("Last complete day", f"{df.index[-1]:%b %d, %Y}")
        c.metric("Model status", "Ready" if bundle else "Ready to train")
    st.caption(
        f"{meta['source']} · {location['timezone']} · {len(df):,} complete days · "
        f"Last complete {df.index[-1].date()}"
    )
    if page == "Historical Weather":
        history_page(df, meta, fahrenheit)
        return
    manager = job_manager()
    token = st.session_state.get("training_job")
    busy = token and manager.snapshot(token)["status"] in ["queued", "running"]
    if page == "Model Lab":
        if st.button(
            "Train new experiment" if bundle else "Train my models", type="primary", disabled=bool(busy)
        ):
            st.session_state["training_job"] = manager.start(df, meta, seed)
            st.rerun()
    if token and (busy or token not in st.session_state.get("finished_jobs", [])):
        job_status(token)
    if bundle is None and page == "Forecast":
        if demo:
            st.info("No synthetic model yet. Open Model Studio to train the demo.")
            st.button("Open Model Studio", on_click=open_model_lab)
        else:
            local_model_empty(location, fahrenheit)
        return
    if bundle is None:
        st.info(
            "Train once to unlock seven-day predictions, uncertainty intervals, and measured accuracy. "
            "The app stays usable while training runs."
        )
        st.markdown(
            '<div class="section-kicker">A look at your starting point / past 60 days</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(temperature_chart(df.tail(60).reset_index(), fahrenheit), width="stretch")
        st.caption("Historical reference data, not predictions. Synthetic when demo is enabled.")
        return
    if page == "Model Lab":
        st.button("View local ML forecast", on_click=open_local_forecast)
        lab(bundle)
        rolling_panel(df, meta, seed)
        return
    try:
        result = forecast(bundle, df, location, demo=demo)
    except ValueError as exc:
        st.warning(str(exc))
        st.info(
            "Historical evaluation remains available in Model Studio. Refresh weather and explicitly retrain for a new origin."
        )
        return
    forecast_page(bundle, result, location, fahrenheit, demo)


def forecast_page(bundle, result, location, fahrenheit, demo):
    unit = "°F" if fahrenheit else "°C"
    first = result.iloc[0]
    high, low = (first.high * 1.8 + 32, first.low * 1.8 + 32) if fahrenheit else (first.high, first.low)
    st.markdown(
        f'<section class="weather-now"><div class="eyebrow">Local ML / experimental daily prediction</div>'
        f'<h2>{escape(location["name"])}</h2><div class="weather-clock">Valid {first.date:%A, %B %d, %Y}</div>'
        f'<div class="weather-reading"><strong>{high:.0f}<small>{unit} high</small></strong></div>'
        f'<div class="weather-condition">Low {low:.0f}{unit} · Rain {first.rain:.0%}</div>'
        '<div class="weather-details">Predicted daily extremes, not a current-temperature reading</div></section>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-kicker">Your seven-day outlook / locally learned</div>', unsafe_allow_html=True
    )
    st.caption(
        f"Origin {bundle['metadata']['origin']} · Model {VERSION} · "
        f"Generated {datetime.now(ZoneInfo(location['timezone'])).isoformat(timespec='seconds')}"
    )
    cards(result, fahrenheit, external=False)
    chart_tab, comparison_tab = st.tabs(["Forecast & uncertainty", "External comparison"])
    with chart_tab:
        st.plotly_chart(temperature_chart(result, fahrenheit, True), width="stretch")
        st.caption(
            "Shaded bands are approximate 80% prediction intervals, estimated on separate calibration data. "
            "They describe plausible daily outcomes, not a guarantee. Wider bands indicate more uncertainty."
        )
    with comparison_tab:
        if demo:
            st.info(
                "External comparison is available in real-data mode. Synthetic and real weather are never overlaid."
            )
        else:
            external_comparison(result, location, fahrenheit)
    st.markdown("#### How does the model hold up?")
    metrics = bundle["metrics"]
    for col, target, label in zip(
        st.columns(3), ["high", "low", "rain"], ["Daily high", "Daily low", "Rain probability"]
    ):
        subset = metrics[metrics.target == target]
        metric = "brier" if target == "rain" else "mae"
        means = subset.groupby("model")[metric].mean()
        ml = means["ML"]
        baseline = means.drop("ML").min()
        delta = (baseline - ml) / baseline * 100 if baseline else 0
        with col:
            st.metric(
                label,
                f"{ml:.3f} Brier" if target == "rain" else f"{ml:.2f}°C MAE",
                f"{abs(delta):.1f}% {'better' if delta >= 0 else 'worse'} than best baseline",
                delta_color="normal" if delta >= 0 else "inverse",
            )
    st.caption(
        "Measured on held-out historical test dates, averaged over seven horizons. "
        "Lower scores are better. These results do not measure the future forecast above."
    )
    with st.expander("Behind this forecast"):
        st.write(
            "Twenty-one local models use lagged observations, trailing averages, and target-date seasonality. "
            "Rain probabilities use a sigmoid calibrator when calibration data supports it. "
            "Temperature bands use seasonal residual quantiles where at least 40 calibration days exist, "
            "otherwise horizon-wide quantiles. Raw and calibrated test scores remain visible in Model Studio."
        )
        st.json(bundle["metadata"])
    st.download_button("Export seven-day forecast", result.to_csv(index=False), "forecast.csv", "text/csv")


def external_comparison(result, location, fahrenheit):
    key = str(location)
    if st.button("Load current external forecast"):
        try:
            payload = request(
                "https://api.open-meteo.com/v1/forecast",
                dict(
                    latitude=location["latitude"],
                    longitude=location["longitude"],
                    timezone=location["timezone"],
                    daily="temperature_2m_max,temperature_2m_min",
                    forecast_days=7,
                    temperature_unit="celsius",
                ),
            )
            daily = payload["daily"]
            external = pd.DataFrame(
                dict(
                    date=pd.to_datetime(daily["time"]),
                    high=daily["temperature_2m_max"],
                    low=daily["temperature_2m_min"],
                )
            )
            if external[["high", "low"]].isna().any().any():
                raise ValueError("Provider returned incomplete temperatures")
            st.session_state["external"] = (
                key,
                external,
                datetime.now(ZoneInfo(location["timezone"])).isoformat(),
            )
        except Exception as exc:
            st.error(f"External forecast unavailable: {exc}")
    stored = st.session_state.get("external")
    if stored and stored[0] == key:
        external = stored[1]
        aligned = result.merge(external, on="date", suffixes=("", "_external"))
        fig = temperature_chart(aligned, fahrenheit)
        for target, color in [("high", "#d8b2ff"), ("low", "#a59dff")]:
            values = aligned[target + "_external"]
            if fahrenheit:
                values = values * 1.8 + 32
            fig.add_trace(
                go.Scatter(
                    x=aligned.date, y=values, name="External " + target, line=dict(color=color, dash="dash")
                )
            )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Solid: local ML. Dashed: current external Open-Meteo operational forecast. "
            f"Matched valid dates only · Retrieved {stored[2]}. This is not a historical accuracy comparison."
        )
    else:
        st.info("Load a separate current operational forecast to compare matching valid dates.")


def experiments_page(location, demo):
    runs = list_runs(location=location, demo=demo)
    st.subheader("Your experiment library")
    if not runs:
        st.info(
            "Train your first experiment on the Forecast page. Every completed run is saved automatically."
        )
        return
    options = [meta["run_id"] for meta in runs]
    selected = st.multiselect("Compare experiments", options, default=options[:2])
    chosen = [m for m in runs if m["run_id"] in selected]
    table = experiment_table(chosen)
    if table.empty:
        return
    st.dataframe(table, hide_index=True, width="stretch")
    target = st.selectbox("Compare target", ["high", "low", "rain"])
    subset = table[(table.target == target) & (table.model == "ML")]
    st.bar_chart(subset.set_index("run").value, y_label="Brier" if target == "rain" else "MAE (°C)")
    windows = {(m["data_fingerprint"], str(m["boundaries"])) for m in chosen}
    if len(windows) > 1:
        st.warning(
            "These experiments use different data or evaluation windows. Scores are contextual, "
            "not a controlled improvement comparison; inspect the dates and fingerprints."
        )
    for meta in chosen:
        with st.expander("Inspect " + meta["run_id"]):
            st.json(meta)
            st.dataframe(get_run(meta["run_id"])["metrics"], hide_index=True)
    st.download_button(
        "Export experiment comparison", table.to_csv(index=False), "experiments.csv", "text/csv"
    )
    with st.expander("Explore other saved locations"):
        all_runs = experiment_table(list_runs(demo=demo))
        st.dataframe(all_runs, hide_index=True)
        st.caption(
            "Each city has its own trained models. Different geographic/date windows are not interchangeable."
        )


def rolling_panel(df, meta, seed):
    st.markdown("### Across time, not just one split")
    st.write(
        "Run three expanding-window backtests with 90 test origins each, 120 calibration origins, "
        "and purged boundaries. Models are refitted independently for each fold. "
        "Later folds can learn earlier folds’ weather, just as time advances."
    )
    manager = job_manager()
    token = st.session_state.get("rolling_job")
    state = manager.snapshot(token) if token else None
    busy = state and state["status"] in ["queued", "running"]
    if st.button("Run rolling backtests", disabled=bool(busy)):
        st.session_state["rolling_job"] = manager.start(df, meta, seed, rolling=True)
        st.session_state["rolling_identity"] = (fingerprint(df), seed, demo_flag(meta))
        st.rerun()
    if token:
        job_status(token)
        state = manager.snapshot(token)
        if state["status"] == "complete":
            if st.session_state.get("rolling_identity") != (fingerprint(df), seed, demo_flag(meta)):
                st.info(
                    "The completed backtest belongs to a different location or mode. Run this location to compare."
                )
                return
            result = state["result"]
            st.dataframe(result["metrics"].dropna(axis=1, how="all"), hide_index=True)
            st.dataframe(
                seasonal_diagnostics(result["predictions"]).dropna(axis=1, how="all"), hide_index=True
            )
            st.json(result["boundaries"])
            st.download_button(
                "Export rolling evaluation",
                result["metrics"].to_csv(index=False),
                "rolling-metrics.csv",
                "text/csv",
            )


def demo_flag(meta):
    return bool(meta.get("synthetic"))


def history_page(df, meta, fahrenheit):
    st.markdown("### The weather behind the model")
    dates = st.date_input(
        "Inspect a range",
        (df.index[0].date(), df.index[-1].date()),
        min_value=df.index[0].date(),
        max_value=df.index[-1].date(),
    )
    if len(dates) != 2:
        return
    view = df.loc[str(dates[0]) : str(dates[1])]
    if view.empty:
        st.info("No days in this range.")
        return
    st.plotly_chart(temperature_chart(view.reset_index(), fahrenheit), width="stretch")
    st.bar_chart(view.rain, y_label="Precipitation (mm)")
    st.markdown("#### Seasonal profile · selected range")
    monthly = view.groupby(view.index.month)[["high", "low"]].mean()
    if fahrenheit:
        monthly = monthly * 1.8 + 32
    st.line_chart(monthly, y_label="°F" if fahrenheit else "°C", x_label="Month")
    a, b, c = st.columns(3)
    a.metric("Complete days", len(view))
    b.metric("Missing cells", int(view.isna().sum().sum()))
    c.metric("Rain total", f"{view.rain.sum():.1f} mm")
    with st.expander("Dataset provenance"):
        st.json(meta)
    st.download_button(
        "Download selected history",
        view.assign(
            temperature_unit="°C",
            precipitation_unit="mm",
            source=meta["source"],
            synthetic=bool(meta.get("synthetic")),
            city=meta["location"]["name"],
            timezone=meta["location"]["timezone"],
        ).to_csv(),
        "history.csv",
        "text/csv",
    )
    st.caption("History exports use °C and mm regardless of display units.")


def lab(bundle):
    st.markdown("### Evidence before confidence")
    st.write(
        "Chronological holdout with seven origin days purged before each boundary. "
        "Calibration determines intervals; the final test set scores all three methods on identical dates. "
        "Models stay fitted to the training partition; inference does not silently refit on the holdout."
    )
    boundaries = bundle["metadata"]["boundaries"]
    st.dataframe(pd.DataFrame(boundaries, index=["First origin", "Last origin"]).T, width="stretch")
    with st.expander("Experiment metadata"):
        st.json(bundle["metadata"])
    metrics = bundle["metrics"]
    target = st.selectbox("Evaluation target", ["high", "low", "rain"])
    subset = metrics[metrics.target == target]
    primary = "brier" if target == "rain" else "mae"
    fig = go.Figure()
    for name, group in subset.groupby("model"):
        fig.add_trace(go.Scatter(x=group.horizon, y=group[primary], name=name, mode="lines+markers"))
    fig.update_layout(
        xaxis_title="Horizon (days)", yaxis_title="Brier score" if target == "rain" else "MAE (°C)"
    )
    st.plotly_chart(fig, width="stretch")
    means = subset.groupby("model")[primary].mean()
    best = means.idxmin()
    st.info(
        f"Lowest overall {primary.upper()}: {best}. Overall values are equal-weight means over seven horizons."
    )
    if best != "ML":
        st.warning("The learned model loses to a baseline on this target in the held-out evaluation.")
    st.dataframe(subset.dropna(axis=1, how="all"), hide_index=True, width="stretch")
    st.dataframe(means.rename("overall").reset_index(), hide_index=True)
    horizon = st.slider("Inspect horizon", 1, 7, 1)
    predictions = bundle["predictions"]
    data = predictions[(predictions.target == target) & (predictions.horizon == horizon)]
    if target != "rain":
        st.line_chart(data.set_index("date")[["actual", "predicted"]], y_label="Temperature (°C)")
        st.caption("Coverage and width columns above refer to nominal 80% intervals. Width is in °C.")
    else:
        bins = pd.cut(data.predicted, np.linspace(0, 1, 6), include_lowest=True)
        reliability = data.groupby(bins, observed=True).agg(
            probability=("predicted", "mean"), frequency=("actual", "mean"), n=("actual", "size")
        )
        if len(data) >= 30:
            fig = go.Figure(
                go.Scatter(
                    x=reliability.probability,
                    y=reliability.frequency,
                    mode="markers+lines",
                    name="Measured frequency",
                    text=reliability.n,
                )
            )
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Ideal", line=dict(dash="dot")))
            fig.update_layout(
                xaxis_title="Predicted rain probability", yaxis_title="Observed event frequency"
            )
            st.plotly_chart(fig, width="stretch")
        st.dataframe(reliability)
        st.caption(
            "Confusion matrix: [[true negative, false positive], [false negative, true positive]]. "
            "Decision threshold 0.5. ROC-AUC is unavailable for one-class test sets."
        )
    with st.expander("Performance by season & horizon"):
        diagnostic = seasonal_diagnostics(bundle["predictions"])
        st.dataframe(diagnostic[diagnostic.target == target].dropna(axis=1, how="all"), hide_index=True)
        st.caption(
            "Seasonal interval quantiles use calibration data only, with a horizon-wide fallback "
            "below 40 calibration days per season. Small test groups are noisy; n is shown."
        )
    if target == "rain":
        rain = metrics[(metrics.target == "rain") & (metrics.model == "ML")]
        st.dataframe(rain[["horizon", "raw_brier", "brier"]], hide_index=True)
        st.caption(
            "Raw vs sigmoid-calibrated Brier score, both measured on untouched test dates. "
            "Calibration can improve or worsen a score; no best-of-test selection is performed."
        )
    st.download_button(
        "Download all evaluation metrics",
        metrics.assign(
            city=bundle["metadata"]["provenance"]["location"]["name"],
            synthetic=bool(bundle["metadata"]["provenance"].get("synthetic")),
            reference_source=bundle["metadata"]["provenance"]["source"],
            temperature_unit="°C",
        ).to_csv(index=False),
        "metrics.csv",
        "text/csv",
    )
    st.info(
        "Archived professional-forecast benchmarking is unavailable in this release. "
        "A current forecast cannot serve as an archived forecast for these test dates."
    )


def about():
    st.markdown("### Weather source & timing")
    st.write(
        "Open-Meteo supplies the current estimates and seven-day operational forecasts shown in Weather "
        "and Compare Cities. These are separate from your locally trained ML models and are not necessarily "
        "station observations. The time beneath current weather is the provider’s valid time in the city’s "
        "local timezone; the date and clock above the temperature show the city’s current local time. "
        "Weather refreshes every ten minutes while the Weather tab is open, and Refresh weather requests "
        "a fresh snapshot. Current estimates more than three hours old are flagged."
    )
    st.write(
        "External daily rain percentages are the maximum hourly precipitation probability during each "
        "local calendar day. Local ML rain percentages instead describe the probability of daily "
        "precipitation reaching 0.1 mm. CSV exports retain source, timezone, retrieval time, and units."
    )
    st.write(
        "The atmospheric background is decorative artwork, not a photograph of the selected city. "
        "Its theme follows the provider’s current weather code and day/night setting. Rain, snow, and "
        "storm effects do not represent measured intensity. Missing or stale readings and synthetic "
        "demo mode use a neutral theme; reduced-motion preferences disable animations."
    )
    st.markdown("### Weather, with its workings visible")
    st.write(
        "Forecast.exe is a Python-first, locally trained weather experiment. "
        "Daily maximum and minimum temperature use direct gradient boosting regressors; "
        "rain occurrence uses classification at ≥0.1 mm. Seven horizons use history ending at the origin. "
        "Persistence uses the origin temperature and trailing 30-day rain frequency. "
        "Seasonal climatology uses training-only reference days within ±15 calendar days."
    )
    st.markdown("#### Reading the evidence")
    st.write(
        "MAE is average absolute temperature error; RMSE emphasizes larger errors. Bias is predicted minus "
        "actual temperature. Brier score is mean squared probability error; lower is better. "
        "ROC-AUC measures event ranking. Interval coverage is the fraction of test outcomes within "
        "the empirical residual bands; it need not reach the nominal 80%. Rain probabilities use "
        "sigmoid calibration on a separate period when both event classes and varied predictions exist."
    )
    st.markdown("#### Sources & limitations")
    st.write(
        "Open-Meteo historical data includes reanalysis: estimates of past weather, not necessarily "
        "direct station measurements. Geographic performance varies, extremes are difficult, and these "
        "statistical models lack the atmospheric physics used by professional medium-range forecasts. "
        "Optional humidity and wind predictors are omitted in this MVP. No weather categories or "
        "precipitation amounts are inferred from the rain probability."
    )
    st.write(
        "Live forecasts require history through yesterday in the selected timezone. Delayed archives "
        "disable live inference; historical analysis remains available. Synthetic mode is always labeled "
        "and produces historical demonstration dates. Artifact files must be generated locally and trusted."
    )
    st.markdown(
        "[Historical API](https://open-meteo.com/en/docs/historical-weather-api) · "
        "[Geocoding](https://open-meteo.com/en/docs/geocoding-api) · "
        "[Terms](https://open-meteo.com/en/terms) · "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"
    )
    st.caption(
        "Free API: noncommercial use only; fewer than 10,000 calls/day, 5,000/hour, 600/minute. "
        "Review current terms before public or commercial deployment. Geocoding: GeoNames via Open-Meteo."
    )
