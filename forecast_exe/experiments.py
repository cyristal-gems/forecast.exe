"""Local experiment registry with immutable runs and cooperative background jobs."""

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from forecast_exe.model import VERSION, fingerprint, load, save, train


def register(bundle, root=Path("artifacts")):
    """Keep every run; JSON summaries are readable without unpickling models."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = root / f"{run_id}.joblib"
    bundle["metadata"]["run_id"] = run_id
    summary = []
    for target, group in bundle["metrics"].groupby("target"):
        metric = "brier" if target == "rain" else "mae"
        for model, values in group.groupby("model"):
            summary.append(
                dict(target=target, model=model, metric=metric, value=float(values[metric].mean()))
            )
    bundle["metadata"]["summary"] = summary
    save(bundle, path)
    return run_id


def list_runs(root=Path("artifacts"), location=None, demo=None):
    rows = []
    for path in sorted(Path(root).glob("*.json"), reverse=True):
        try:
            meta = json.loads(path.read_text())
            if meta.get("version") != VERSION or not meta.get("run_id"):
                continue
            required = {
                "data_fingerprint",
                "seed",
                "summary",
                "boundaries",
                "origin",
                "created_at",
                "provenance",
            }
            if not required.issubset(meta) or path.stem != meta["run_id"]:
                continue
            provenance = meta["provenance"]
            if location is not None and provenance["location"] != location:
                continue
            if demo is not None and bool(provenance.get("synthetic")) != demo:
                continue
            if not path.with_suffix(".joblib").exists():
                continue
            rows.append(meta)
        except (ValueError, KeyError, OSError):
            continue
    return rows


def compatible_runs(df, location, demo, seed=42, root=Path("artifacts")):
    digest = fingerprint(df)
    return [
        m for m in list_runs(root, location, demo) if m["data_fingerprint"] == digest and m["seed"] == seed
    ]


def experiment_table(runs):
    rows = []
    for meta in runs:
        for score in meta["summary"]:
            rows.append(
                dict(
                    run=meta["run_id"],
                    city=meta["provenance"]["location"]["name"],
                    seed=meta["seed"],
                    synthetic=bool(meta["provenance"].get("synthetic")),
                    reference_source=meta["provenance"]["source"],
                    timezone=meta["provenance"]["location"]["timezone"],
                    origin=meta["origin"],
                    created=meta["created_at"],
                    fingerprint=meta["data_fingerprint"][:12],
                    test_start=meta["boundaries"]["test_origins"][0],
                    test_end=meta["boundaries"]["test_origins"][1],
                    **score,
                )
            )
    return pd.DataFrame(rows)


class TrainingCancelled(Exception):
    """Cancellation is observed between horizon fits."""


class Jobs:
    """In-process single-worker queue: no Streamlit API calls in worker threads."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecast-training")
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, df, meta, seed=42, root=Path("artifacts"), rolling=False):
        token = uuid.uuid4().hex
        state = dict(progress=0.0, status="queued", error=None, run=None, cancel=threading.Event())
        with self.lock:
            self.jobs[token] = state

        def update(value):
            if state["cancel"].is_set():
                raise TrainingCancelled()
            with self.lock:
                state["progress"] = value

        def work():
            try:
                state["status"] = "running"
                if rolling:
                    from forecast_exe.evaluation import rolling_backtest

                    result = rolling_backtest(df, meta, seed=seed, progress=update)
                    state["result"] = result
                else:
                    bundle = train(df.copy(), meta, seed, progress=update)
                    update(1.0)
                    state["run"] = register(bundle, root)
                state["status"] = "complete"
            except TrainingCancelled:
                state["status"] = "cancelled"
            except Exception as exc:
                state["error"] = str(exc)
                state["status"] = "failed"

        self.executor.submit(work)
        return token

    def snapshot(self, token):
        with self.lock:
            return dict(self.jobs[token])

    def cancel(self, token):
        self.jobs[token]["cancel"].set()


def get_run(run_id, root=Path("artifacts")):
    """Run identifiers come from registry choices, never arbitrary upload paths."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Invalid experiment identifier")
    return load(Path(root) / f"{run_id}.joblib")
