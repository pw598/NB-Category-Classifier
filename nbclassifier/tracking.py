"""MLflow tracking that can be switched off with one flag.

Every notebook creates `tracker = Tracker(USE_MLFLOW, ...)` and calls its
methods unconditionally. With USE_MLFLOW = False (or mlflow not installed)
every call is a silent no-op, so notebook code never needs `if` guards.

On Databricks the default experiment is the notebook's own experiment; set
MLFLOW_EXPERIMENT to a workspace path (e.g. '/Users/you@vallen.com/category-classifier')
to collect runs from all four notebooks in one place.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, enabled: bool, experiment: Optional[str] = None,
                 run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        self.enabled = bool(enabled)
        self.mlflow = None
        self.run = None
        try:
            import mlflow
        except ImportError:
            mlflow = None
        if mlflow is not None:
            # Databricks enables sklearn autologging by default, which would log a run for
            # every vectorizer fit - even with USE_MLFLOW = False. Always switch it off.
            try:
                mlflow.autolog(disable=True)
            except Exception:
                pass
        if not self.enabled:
            return
        if mlflow is None:
            print("[mlflow] not installed; tracking disabled")
            self.enabled = False
            return
        self.mlflow = mlflow
        if experiment:
            mlflow.set_experiment(experiment)
        if mlflow.active_run() is not None:  # setup cell re-run: close the previous run first
            mlflow.end_run()
        self.run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        print(f"[mlflow] run {self.run.info.run_id} ({run_name})")

    # ------------------------------------------------------------------
    def log_params(self, params: Dict[str, Any], prefix: str = "") -> None:
        if not self.enabled:
            return
        flat = {k: ("None" if v is None else str(v))[:500]
                for k, v in _flatten(params, prefix).items()}
        items = list(flat.items())
        for i in range(0, len(items), 100):  # MLflow batch limit
            try:
                self.mlflow.log_params(dict(items[i:i + 100]))
            except Exception as exc:  # e.g. a cell re-run with a changed value (params are immutable)
                print(f"[mlflow] some params not logged: {str(exc).splitlines()[0][:200]}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
        if not self.enabled:
            return
        clean = {}
        for k, v in _flatten(metrics, prefix).items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:  # not NaN
                clean[_safe_key(k)] = f
        if clean:
            self.mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if self.enabled:
            self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_dict(self, obj: Dict[str, Any], artifact_file: str) -> None:
        if self.enabled:
            self.mlflow.log_dict(json.loads(json.dumps(obj, default=str)), artifact_file)

    def log_artifact(self, path, artifact_path: Optional[str] = None) -> None:
        if self.enabled and Path(path).exists():
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, folder, artifact_path: Optional[str] = None) -> None:
        if self.enabled and Path(folder).is_dir():
            self.mlflow.log_artifacts(str(folder), artifact_path=artifact_path)

    def log_figure(self, fig, artifact_file: str) -> None:
        if self.enabled:
            self.mlflow.log_figure(fig, artifact_file)

    def log_table(self, df, artifact_file: str) -> None:
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / Path(artifact_file).name
            df.to_csv(p, index=False)
            parent = str(Path(artifact_file).parent)
            self.mlflow.log_artifact(str(p), artifact_path=None if parent == "." else parent)

    @contextmanager
    def child(self, run_name: str) -> Iterator["Tracker"]:
        """Nested run (e.g. one per grid candidate)."""
        if not self.enabled:
            yield self
            return
        with self.mlflow.start_run(run_name=run_name, nested=True):
            yield self

    def end(self) -> None:
        if self.enabled and self.mlflow.active_run() is not None:
            self.mlflow.end_run()
            print("[mlflow] run ended")


def _safe_key(k: str) -> str:
    # MLflow metric names: alphanumerics, _ - . space / only
    return "".join(ch if (ch.isalnum() or ch in "_-. /") else "_" for ch in k)
