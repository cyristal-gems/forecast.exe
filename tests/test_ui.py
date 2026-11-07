"""Start each UI view without a network connection."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_pages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.session_state["demo_mode"] = True
    app.run()
    assert not app.exception
    assert app.title[0].value == "Forecast.exe"
    for page in ["Historical Weather", "Model Lab", "Experiments", "Methodology"]:
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
