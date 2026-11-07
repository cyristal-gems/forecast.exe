"""Offline location/privacy behavior and independently sourced external weather."""

import copy
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from forecast_exe.data import NYC
from forecast_exe.locations import coordinates, nearby_city, validate_location
from forecast_exe.location_ui import process_event, select_location
from forecast_exe import weather


def test_offline_city_state_country_and_timezone():
    result = nearby_city(40.71, -74.0)
    assert "New York" in result["name"] and "United States" in result["name"]
    assert result["timezone"] == "America/New_York"
    assert (result["latitude"], result["longitude"]) != (40.71, -74.0)
    london = nearby_city(51.51, -0.13)
    assert "England" in london["name"] and "United Kingdom" in london["name"]
    assert london["timezone"] == "Europe/London"


@pytest.mark.parametrize("lat,lon", [(91, 0), (0, -181), (np.nan, 1), (True, 2), ("bad", 2)])
def test_invalid_device_coordinates(lat, lon):
    with pytest.raises(ValueError):
        coordinates(lat, lon)


def test_location_preferences_and_denial_keep_manual_fallback():
    state = {"dataset": "old snapshot"}
    select_location(state, NYC)
    assert "dataset" not in state
    assert state["preference_write"]["city"] == NYC
    assert state["demo_mode"] is False
    assert process_event(state, {"id": "one", "kind": "error", "message": "Permission declined"})
    assert state["active_location"] == NYC
    assert "declined" in state["location_error"]
    assert process_event(
        state,
        {"id": "two", "kind": "detected", "latitude": 10.0, "longitude": 20.0, "remember": True},
        resolver=lambda lat, lon: NYC,
    )
    assert state["preference_write"]["city"]["latitude"] == NYC["latitude"]
    assert not process_event(state, {"id": "two", "kind": "detected"})
    assert process_event(state, {"id": "three", "kind": "forget"})
    assert "active_location" not in state and state["preference_write"]["city"] is None
    assert process_event(state, {"id": "four", "kind": "restore", "city": NYC})
    assert state["active_location"] == NYC
    assert process_event(state, {"id": "five", "kind": "remember", "remember": False})
    assert state["preference_write"]["remember"] is False


def test_saved_preference_validation():
    with pytest.raises(ValueError):
        validate_location(dict(NYC, timezone="Invalid/Zone"))
    with pytest.raises(ValueError):
        validate_location(dict(NYC, name="x" * 200))
    assert process_event({}, {"id": "invalid", "kind": "restore", "city": {"name": "bad"}})


def payload():
    return dict(
        current=dict(
            time="2026-11-07T11:15",
            temperature_2m=16.0,
            apparent_temperature=15.0,
            relative_humidity_2m=65.0,
            weather_code=2,
            wind_speed_10m=12.0,
            is_day=1,
        ),
        daily=dict(
            time=pd.date_range("2026-11-07", periods=7).strftime("%Y-%m-%d").tolist(),
            temperature_2m_max=[19.0] * 7,
            temperature_2m_min=[10.0] * 7,
            precipitation_probability_max=[25.0] * 7,
        ),
        daily_units={
            "temperature_2m_max": "°C",
            "temperature_2m_min": "°C",
            "precipitation_probability_max": "%",
        },
        current_units={
            "temperature_2m": "°C",
            "apparent_temperature": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "km/h",
        },
        utc_offset_seconds=-18000,
        timezone="America/New_York",
    )


def test_current_weather_is_external_and_validated(monkeypatch):
    value = payload()
    calls = []

    def fake(url, params, **kwargs):
        calls.append(params)
        return value

    monkeypatch.setattr(weather, "request", fake)
    result = weather.current_weather(NYC)
    assert "external" in result["source"]
    assert len(result["daily"]) == 7 and (result["daily"].rain == 0.25).all()
    assert calls[0]["current"].startswith("temperature_2m")
    assert calls[0]["latitude"] == NYC["latitude"]
    assert weather.condition(0, False)[0] == "Clear sky"
    invalid = copy.deepcopy(value)
    invalid["daily"]["precipitation_probability_max"][0] = 101
    monkeypatch.setattr(weather, "request", lambda *a, **k: invalid)
    with pytest.raises(ValueError):
        weather.current_weather(NYC)


def test_current_weather_without_local_models(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    from forecast_exe.weather_ui import weather_snapshot

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(weather, "request", lambda *a, **k: payload())
    weather_snapshot.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90)
    app.session_state["city_preferences_ready"] = True
    app.run()
    assert not app.exception
    assert any("External weather / Open-Meteo" in node.value for node in app.markdown)
    assert not (tmp_path / "artifacts").exists()
    source = next(r for r in app.radio if r.label == "Forecast source")
    source.set_value("Local ML").run()
    assert not app.exception
    assert any("not ready" in node.value for node in app.subheader)
    assert any("External weather / Open-Meteo" in node.value for node in app.markdown)
    weather_snapshot.clear()


def test_browser_location_component_is_permission_driven():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed only for the frontend unit checks")
    result = subprocess.run(
        [node, str(Path(__file__).with_name("test_location_bridge.mjs"))],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_header_navigation_and_cinematic_branding(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90)
    app.session_state["demo_mode"] = True
    app.run()
    assert not app.exception
    assert len(app.sidebar) == 0
    labels = [tab.label for tab in app.tabs]
    assert labels[:5] == ["Weather", "Weather History", "Model Studio", "Compare Cities", "How It Works"]
    content = " ".join(node.value for node in app.markdown)
    assert "Decoding the Skies, One Prediction at a Time." in content
    assert 'aria-label="Reload Forecast.exe"' in content
    assert 'href="./"' in content
    assert "THE INDEPENDENT WEATHER STUDIO" not in content
    for page in ["Weather History", "Compare Cities", "How It Works", "Model Studio"]:
        app.session_state["workspace"] = page
        app.run()
        assert not app.exception


def test_city_comparison_keeps_local_dates_and_sources(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    from forecast_exe.comparisons import LONDON, add_to_comparison, comparison_frame
    from forecast_exe.weather_ui import weather_snapshot

    monkeypatch.chdir(tmp_path)
    state = {}
    add_to_comparison(state, LONDON)
    assert state["workspace"] == "Compare Cities"
    assert len(state["compare_city_ids"]) == 2
    rows = comparison_frame(
        [
            (
                NYC,
                {
                    "daily": pd.DataFrame(
                        dict(date=pd.to_datetime(["2026-11-07"]), high=[19.0], low=[10.0], rain=[0.2])
                    )
                },
            ),
            (
                LONDON,
                {
                    "daily": pd.DataFrame(
                        dict(date=pd.to_datetime(["2026-11-07"]), high=[12.0], low=[6.0], rain=[0.4])
                    )
                },
            ),
        ]
    )
    assert set(rows.timezone) == {"America/New_York", "Europe/London"}
    assert rows.source.str.contains("external").all()
    monkeypatch.setattr(
        weather,
        "request",
        lambda url, params, **kw: dict(
            payload(),
            timezone=params["timezone"],
            utc_offset_seconds=0 if params["timezone"] == "Europe/London" else -18000,
        ),
    )
    weather_snapshot.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90)
    app.session_state["workspace"] = "Compare Cities"
    app.session_state["city_preferences_ready"] = True
    app.run()
    assert not app.exception
    assert len(app.get("plotly_chart")) >= 1
    import json

    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["showlegend"] is False
    assert spec["layout"]["margin"]["b"] >= 90
    assert any("chart-key" in node.value for node in app.markdown)
    weather_snapshot.clear()


def test_each_tab_has_five_sentence_summary():
    from forecast_exe.ui import TAB_SUMMARIES

    assert len(TAB_SUMMARIES) == 5
    for summary in TAB_SUMMARIES.values():
        assert summary.count(".") == 5


def test_external_units_timezone_and_exports_cannot_be_mislabelled(monkeypatch):
    value = payload()
    monkeypatch.setattr(weather, "request", lambda *args, **kwargs: value)
    result = weather.current_weather(NYC)
    assert result["daily"].temperature_unit.eq("°C").all()
    assert result["daily"].source.str.contains("external").all()
    assert result["daily"].rain_definition.str.contains("hourly").all()
    assert result["daily"].timezone.eq("America/New_York").all()
    value["current_units"]["temperature_2m"] = "°F"
    with pytest.raises(ValueError, match="units"):
        weather.current_weather(NYC)
    value["current_units"]["temperature_2m"] = "°C"
    value["utc_offset_seconds"] = 0
    with pytest.raises(ValueError, match="offset"):
        weather.current_weather(NYC)


@pytest.mark.parametrize(
    "code,day,expected",
    [
        (0, True, "sunny"),
        (0, False, "clear-night"),
        (3, True, "cloudy"),
        (3, False, "cloudy-night"),
        (61, True, "rain"),
        (95, False, "storm"),
        (71, True, "snow"),
        (45, True, "fog"),
        (999, True, "neutral"),
    ],
)
def test_weather_background_follows_provider_codes(code, day, expected):
    from forecast_exe.themes import weather_theme, theme_css

    assert weather_theme(code, day) == expected
    assert "prefers-reduced-motion" in theme_css(expected)


def test_background_does_not_invent_weather_for_other_cities_or_demo(monkeypatch):
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace
    import forecast_exe.themes as themes

    state = {}
    monkeypatch.setattr(themes, "st", SimpleNamespace(session_state=state, markdown=lambda *a, **kw: None))
    snapshot = dict(valid_at_utc=datetime.now(UTC).isoformat(), current={"weather_code": 61, "is_day": 1})
    assert themes.apply_background(NYC, snapshot=snapshot) == "rain"
    assert themes.apply_background(NYC, demo=True) == "neutral"
    assert themes.apply_background(dict(NYC, latitude=20)) == "neutral"
    state["weather_theme"]["valid_at_utc"] = (datetime.now(UTC) - timedelta(hours=4)).isoformat()
    assert themes.apply_background(NYC) == "neutral"


def test_contacts_are_clickable_icons_not_visible_urls():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string("from forecast_exe.branding import contact_dock\ncontact_dock()").run()
    assert not app.exception
    html = app.markdown[0].value
    assert 'aria-label="LinkedIn profile"' in html and "https://www.linkedin.com/in/cyristalj" in html
    assert "https://github.com/cyristal-gems" in html
    assert "mailto:cyrisjoseph@outlook.com" in html
    assert html.count("<svg") == 3
    assert ">https://" not in html and ">cyrisjoseph@outlook.com" not in html
