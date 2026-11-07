"""Streamlit entrypoint for Forecast.exe."""

from pathlib import Path
from forecast_exe.ui import main

main(asset_root=Path(__file__).resolve().parent / "static")
