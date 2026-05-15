"""
src/mlops/experiment.py
========================
MLflow experiment tracking helpers.

Wraps MLflow run management so the rest of the pipeline
can log params, metrics, and artifacts with minimal boilerplate.
"""

from __future__ import annotations
import os
import logging
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning("mlflow not installed. Experiment tracking disabled. Run: pip install mlflow")


def setup_mlflow(
    experiment_name: str = "forex-ensemble-prediction",
    tracking_uri: str | None = None,
) -> None:
    """
    Configure MLflow tracking URI and experiment.

    Parameters
    ----------
    experiment_name : MLflow experiment name to log runs under.
    tracking_uri    : MLflow tracking server URI.
                      Defaults to the MLFLOW_TRACKING_URI env var,
                      or local './mlruns' if not set.
    """
    if not MLFLOW_AVAILABLE:
        return
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    log.info(f"MLflow tracking URI: {uri}  |  Experiment: {experiment_name}")


@contextmanager
def start_run(run_name: str, tags: dict | None = None):
    """
    Context manager that starts an MLflow run and ensures it ends cleanly.

    Usage
    -----
    with start_run("xgboost_baseline") as run:
        log_params({"lr": 0.05})
        log_metrics({"MAE": 0.012})

    Parameters
    ----------
    run_name : Display name for this MLflow run.
    tags     : Optional dict of string tags to attach.
    """
    if not MLFLOW_AVAILABLE:
        yield None
        return
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        log.info(f"MLflow run started: {run.info.run_id}  [{run_name}]")
        yield run
    log.info(f"MLflow run ended: {run.info.run_id}")


def log_params(params: dict[str, Any]) -> None:
    """Log a dict of hyperparameters to the active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of metrics to the active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    """Log a file or directory as an MLflow artifact."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow.log_artifact(path)
    log.info(f"MLflow artifact logged: {path}")


def log_model_artifact(model_path: str, artifact_path: str = "models") -> None:
    """Log a serialized model file as an MLflow artifact."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow.log_artifact(model_path, artifact_path=artifact_path)
    log.info(f"MLflow model artifact logged: {model_path} -> {artifact_path}")
