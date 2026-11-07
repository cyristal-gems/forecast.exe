"""Release checks for incremental data, immutable runs, calibration and backtests."""

import threading
import time

import numpy as np
import pandas as pd
import pytest

from forecast_exe import data
from forecast_exe.evaluation import rolling_splits, seasonal_diagnostics
from forecast_exe.experiments import Jobs, compatible_runs, list_runs, register
from forecast_exe.model import calibrate, fingerprint, fit_calibrator, forecast, load, train


@pytest.fixture(scope="module")
def release_model():
    df, meta = data.synthetic()
    return df, meta, train(df, meta)


def test_incremental_requests_and_failure_preserves_master(tmp_path, monkeypatch):
    source, _ = data.synthetic()
    calls = []

    def fake(location, start, end, root, refresh=False):
        calls.append((pd.Timestamp(start), pd.Timestamp(end), refresh))
        return source.loc[str(start) : str(end)], dict(
            location=location, source="mock", retrieved_at="fixture"
        )

    monkeypatch.setattr(data, "history", fake)
    first, _ = data.update_history(data.NYC, "2019-01-01", "2019-01-10", tmp_path)
    assert len(first) == 10
    second, _ = data.update_history(data.NYC, "2019-01-01", "2019-01-12", tmp_path)
    assert calls[-1][:2] == (pd.Timestamp("2019-01-11"), pd.Timestamp("2019-01-12"))
    data.update_history(data.NYC, "2019-01-01", "2019-01-12", tmp_path)
    assert len(calls) == 2
    refreshed, _ = data.update_history(data.NYC, "2019-01-01", "2019-01-12", tmp_path, refresh=True)
    assert calls[-1] == (pd.Timestamp("2019-01-06"), pd.Timestamp("2019-01-12"), True)
    pd.testing.assert_frame_equal(second, refreshed)
    master = next(tmp_path.glob("location-*.parquet"))
    before = master.read_bytes()

    def fail(*args, **kwargs):
        raise ValueError("Unavailable source")

    monkeypatch.setattr(data, "history", fail)
    with pytest.raises(ValueError, match="Unavailable"):
        data.update_history(data.NYC, "2019-01-01", "2019-01-13", tmp_path)
    assert master.read_bytes() == before


def test_content_fingerprints_and_immutable_runs(release_model, tmp_path):
    df, meta, bundle = release_model
    first = register(bundle, tmp_path)
    second = register(bundle, tmp_path)
    assert first != second and len(list_runs(tmp_path)) == 2
    assert len(compatible_runs(df, data.NYC, True, root=tmp_path)) == 2
    changed = df.copy()
    changed.iloc[300, 0] += 0.01
    assert fingerprint(df) != fingerprint(changed)
    assert compatible_runs(changed, data.NYC, True, root=tmp_path) == []
    with pytest.raises(ValueError, match="fingerprint"):
        forecast(bundle, changed, data.NYC, demo=True)
    restored = load(tmp_path / f"{first}.joblib")
    assert restored["metadata"]["run_id"] == first


def test_calibration_and_seasonal_coverage(release_model):
    df, meta, bundle = release_model
    assert fit_calibrator(np.ones(50) * 0.4, np.zeros(50)) is None
    np.testing.assert_array_equal(calibrate(None, [0.1, 0.9]), [0.1, 0.9])
    assert len(bundle["calibrators"]) == 7
    assert len(bundle["seasonal_intervals"]) == 56
    rain = bundle["metrics"].query("target == 'rain' and model == 'ML'")
    assert rain.raw_brier.notna().all()
    diagnostic = seasonal_diagnostics(bundle["predictions"])
    assert diagnostic.query("model == 'ML' and target != 'rain'").coverage.between(0, 1).all()
    assert diagnostic.n.min() > 0
    assert set(diagnostic.model) == {"ML", "Persistence", "Climatology"}


def test_rolling_splits_are_purged_and_expand():
    df, _ = data.synthetic()
    splits = rolling_splits(df.index[29:-7])
    for fit, cal, test in splits:
        assert fit.max() + pd.Timedelta(days=7) < cal.min()
        assert cal.max() + pd.Timedelta(days=7) < test.min()
        assert len(test) == 90 and len(cal) == 120
    assert splits[0][2].max() < splits[1][2].min()
    assert len(splits[0][0]) < len(splits[1][0]) < len(splits[2][0])
    with pytest.raises(ValueError, match="Insufficient"):
        rolling_splits(df.index[:730])


def test_cancel_background_training_saves_no_partial(monkeypatch, tmp_path):
    started = threading.Event()
    gate = threading.Event()

    def fake_train(df, meta, seed, progress):
        started.set()
        gate.wait(5)
        progress(0.1)
        raise AssertionError("Cancelled job should stop at progress callback")

    monkeypatch.setattr("forecast_exe.experiments.train", fake_train)
    jobs = Jobs()
    df, meta = data.synthetic()
    token = jobs.start(df, meta, root=tmp_path)
    assert started.wait(5)
    jobs.cancel(token)
    gate.set()
    for _ in range(100):
        if jobs.snapshot(token)["status"] == "cancelled":
            break
        time.sleep(0.01)
    assert jobs.snapshot(token)["status"] == "cancelled"
    assert list(tmp_path.iterdir()) == []
    jobs.executor.shutdown(wait=True)


def test_incomplete_artifact_rejected(release_model, tmp_path):
    import joblib

    df, meta, bundle = release_model
    broken = dict(bundle, calibrators=dict(bundle["calibrators"]))
    broken["calibrators"][1] = "invalid"
    path = tmp_path / "broken.joblib"
    joblib.dump(broken, path)
    with pytest.raises(ValueError, match="Invalid probability"):
        load(path)
    broken = dict(bundle, models=dict(bundle["models"]))
    broken["models"].pop((7, "rain"))
    joblib.dump(broken, path)
    with pytest.raises(ValueError, match="Incomplete horizon"):
        load(path)


def test_ready_interface_and_rain_diagnostics(release_model, monkeypatch, tmp_path):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    df, meta, bundle = release_model
    register(bundle, tmp_path / "artifacts")
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90)
    app.session_state["demo_mode"] = True
    app.run()
    assert not app.exception
    assert any("Rain" in node.value for node in app.markdown)
    for page in ["Model Lab", "Experiments", "Historical Weather", "Methodology"]:
        app.session_state["workspace"] = {
            "Model Lab": "Model Studio",
            "Experiments": "Model Studio",
            "Historical Weather": "Weather History",
            "Methodology": "How It Works",
        }[page]
        if page == "Experiments":
            app.session_state["studio_section"] = "Saved Runs"
        elif page == "Model Lab":
            app.session_state["studio_section"] = "Train & evaluate"
        app.run()
        assert not app.exception
        if page == "Model Lab":
            target = next(box for box in app.selectbox if box.label == "Evaluation target")
            target.set_value("rain").run()
            assert not app.exception


def test_external_overlay_keeps_separate_matched_series(monkeypatch):
    import json
    from streamlit.testing.v1 import AppTest
    import forecast_exe.ui as ui

    dates = pd.date_range("2020-01-01", periods=8)
    monkeypatch.setattr(
        ui,
        "request",
        lambda *a: {
            "daily": {
                "time": dates.strftime("%Y-%m-%d").tolist(),
                "temperature_2m_max": [12.0] * 8,
                "temperature_2m_min": [2.0] * 8,
            }
        },
    )
    script = """
import pandas as pd
from forecast_exe.ui import external_comparison
from forecast_exe.data import NYC
result = pd.DataFrame(dict(date=pd.date_range('2020-01-01', periods=7), high=[10.]*7, low=[1.]*7))
external_comparison(result, NYC, False)
"""
    app = AppTest.from_string(script, default_timeout=90).run()
    app.button[0].click().run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert {trace["name"] for trace in spec["data"]} == {"High", "Low", "External high", "External low"}
    assert all(len(trace["x"]) == 7 for trace in spec["data"])
