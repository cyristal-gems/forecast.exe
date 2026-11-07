"""Install a locked local environment and open Forecast.exe. Python 3.11+ required."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent


def environment_python():
    return ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    if sys.version_info < (3, 11):
        print("Forecast.exe needs Python 3.11 or newer. Install it from https://www.python.org/downloads/.")
        return 1
    python = environment_python()
    lock = ROOT / "requirements.lock"
    identity = hashlib.sha256(lock.read_bytes() + (ROOT / "pyproject.toml").read_bytes()).hexdigest()
    stamp = ROOT / ".venv/.forecast-install-id"
    try:
        if not python.exists():
            print("Creating your local Forecast.exe environment…")
            venv.create(ROOT / ".venv", with_pip=True)
        if not stamp.exists() or stamp.read_text() != identity:
            print("Installing pinned dependencies. The first launch needs an internet connection…")
            subprocess.run([str(python), "-m", "pip", "install", "-r", str(lock)], check=True, cwd=ROOT)
            subprocess.run(
                [str(python), "-m", "pip", "install", "--no-deps", "-e", "."], check=True, cwd=ROOT
            )
            stamp.write_text(identity)
        if "--setup-only" in sys.argv:
            print("Forecast.exe is installed and ready. Run this launcher again to open the app.")
            return 0
        url = "http://127.0.0.1:8501"
        print(f"Opening Forecast.exe at {url}. Keep this window open; press Ctrl+C to stop.")
        stopped = threading.Event()

        def open_when_ready():
            for _ in range(60):
                if stopped.is_set():
                    return
                try:
                    with urllib.request.urlopen(url + "/_stcore/health", timeout=1) as response:
                        if response.status == 200:
                            webbrowser.open(url)
                            return
                except OSError:
                    time.sleep(0.5)

        command = [
            str(python),
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ]
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
        threading.Thread(target=open_when_ready, daemon=True).start()
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            return 0
        finally:
            stopped.set()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Launch failed: {exc}. Check your internet connection and Python installation, then retry.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
