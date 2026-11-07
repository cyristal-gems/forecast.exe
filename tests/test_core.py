"""Offline tests use explicitly synthetic data and mocked provider responses."""

import numpy as np
import pandas as pd
import pytest

from forecast_exe import data
from forecast_exe.model import features, forecast, load, partitions, save, scores, train


@pytest.fixture(scope="module")
def fitted():
    df, meta = data.synthetic()
    return df, meta, train(df, meta)


def test_origin_features_are_unchanged_by_future():
    df, _ = data.synthetic()
    t = df.index[500]
    changed = df.copy()
    changed.loc[changed.index > t] += 100
    for h in range(1, 8):
        pd.testing.assert_series_equal(features(df, h).loc[t], features(changed, h).loc[t])


def test_split_purges_targets():
    index = pd.date_range("2020-01-01", periods=1000)
    a, b, c = partitions(index)
    assert a.max() + pd.Timedelta(days=7) < b.min()
    assert b.max() + pd.Timedelta(days=7) < c.min()


def test_validation():
    df, _ = data.synthetic()
    for invalid in [df.iloc[::-1], df.drop(df.index[50]), pd.concat([df, df.iloc[[-1]]])]:
        with pytest.raises(ValueError):
            data.validate(invalid)
    df.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        data.validate(df)


def test_metrics():
    result = scores([1, 3], [2, 1])
    assert result["mae"] == 1.5
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert scores([], []) == {"n": 0}
    assert scores([0, 0], [0.2, 0.2], True)["auc"] is None
    assert scores([0, 1], [0.2, 0.8], True)["brier"] == pytest.approx(0.04)


def test_end_to_end(fitted, tmp_path):
    df, meta, bundle = fitted
    result = forecast(bundle, df, data.NYC, demo=True)
    assert result.horizon.tolist() == list(range(1, 8))
    assert (result.low <= result.high).all()
    assert result.rain.between(0, 1).all()
    assert result.synthetic.eq(True).all()
    assert result.forecast_source.eq("Locally trained ML").all()
    assert result.temperature_unit.eq("°C").all()
    assert np.isfinite(result.select_dtypes("number")).all().all()
    save(bundle, tmp_path / "model.joblib")
    loaded = load(tmp_path / "model.joblib")
    pd.testing.assert_frame_equal(result, forecast(loaded, df, data.NYC, demo=True))
    assert loaded["metadata"]["provenance"] == meta
    metrics = bundle["metrics"]
    assert (metrics.groupby(["target", "horizon"]).n.nunique() == 1).all()
    assert len(metrics) == 63
    assert metrics.query("model == 'ML' and target != 'rain'").coverage.between(0, 1).all()


def test_freshness_and_location(fitted):
    df, _, bundle = fitted
    with pytest.raises(ValueError, match="different location"):
        forecast(bundle, df, dict(data.NYC, name="Elsewhere"), demo=True)
    real = dict(
        bundle,
        metadata=dict(bundle["metadata"], provenance=dict(bundle["metadata"]["provenance"], synthetic=False)),
    )
    with pytest.raises(ValueError, match="Live forecast unavailable"):
        forecast(real, df, data.NYC, now="2026-11-07")
    with pytest.raises(ValueError, match="snapshot changed"):
        forecast(bundle, df.iloc[:-1], data.NYC, demo=True)


def test_reproducibility(fitted):
    df, meta, first = fitted
    second = train(df, meta)
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])


def test_history_and_cache(monkeypatch, tmp_path):
    calls = []
    payload = {
        "daily": {
            "time": ["2020-01-01", "2020-01-02"],
            "temperature_2m_max": [10, 12],
            "temperature_2m_min": [1, 2],
            "precipitation_sum": [0, 1],
        },
        "daily_units": {"temperature_2m_max": "°C", "temperature_2m_min": "°C", "precipitation_sum": "mm"},
    }

    def fake(url, params):
        calls.append(params)
        return payload

    monkeypatch.setattr(data, "request", fake)
    a, meta = data.history(data.NYC, "2020-01-01", "2020-01-02", tmp_path)
    b, _ = data.history(data.NYC, "2020-01-01", "2020-01-02", tmp_path)
    pd.testing.assert_frame_equal(a, b)
    assert len(calls) == 1
    assert meta["request"]["timezone"] == "America/New_York"
    payload["daily"].pop("precipitation_sum")
    with pytest.raises(ValueError, match="omitted"):
        data.history(data.NYC, "2020-01-01", "2020-01-03", tmp_path)


def test_geocoding(monkeypatch):
    monkeypatch.setattr(data, "request", lambda *a: {"results": [data.NYC]})
    assert data.geocode("New York")[0]["latitude"] == data.NYC["latitude"]
    monkeypatch.setattr(data, "request", lambda *a: {})
    assert data.geocode("missing") == []


def test_http_failures(monkeypatch):
    import httpx

    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    for error in [
        httpx.ReadTimeout("timeout"),
        httpx.HTTPStatusError(
            "server", request=httpx.Request("GET", "https://example.com"), response=httpx.Response(503)
        ),
    ]:

        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr(data.httpx, "get", fail)
        with pytest.raises(type(error)):
            data.request("https://example.com", {})

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": True}

    monkeypatch.setattr(data.httpx, "get", lambda *a, **kw: Response())
    with pytest.raises(ValueError):
        data.request("https://example.com", {})
