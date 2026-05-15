"""
src/mlops/train_with_tracking.py
=================================
Dropped-in replacement for the tree model training step in main.py
that wraps training in an MLflow run for full experiment tracking.

Logs: hyperparameters, train/val metrics, model artifacts, SHAP plot.

Usage
-----
Called from main.py when --track flag is enabled, or used standalone:
    python -m src.mlops.train_with_tracking
"""

from __future__ import annotations
import logging
import os
import numpy as np

from src.mlops.experiment import setup_mlflow, start_run, log_params, log_metrics, log_artifact, log_model_artifact
from src.mlops.versioning import make_run_id, save_versioned_model, write_run_manifest
from src.models.tree_models import build_xgboost, build_lightgbm, train_xgboost, train_lightgbm
from src.evaluation.metrics import compute_metrics

log = logging.getLogger(__name__)


def train_trees_with_tracking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y,
    output_dir: str,
    xgb_params: dict | None = None,
    lgb_params: dict | None = None,
) -> tuple:
    """
    Train XGBoost and LightGBM with full MLflow experiment tracking.

    For each model:
      - Logs hyperparameters
      - Logs MAE, RMSE, MAPE, R2, DA metrics
      - Saves versioned + latest model artifacts
      - Writes a run manifest JSON

    Parameters
    ----------
    X_train, y_train : Training features and scaled targets.
    X_test,  y_test  : Test features and scaled targets.
    scaler_y         : Fitted MinMaxScaler for target inverse-transform.
    output_dir       : Root output directory.
    xgb_params       : Optional XGBoost hyperparameter overrides.
    lgb_params       : Optional LightGBM hyperparameter overrides.

    Returns
    -------
    (xgb_model, lgb_model, run_id)
    """
    run_id = make_run_id()
    setup_mlflow()

    def inv(arr):
        return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb_model = build_xgboost(xgb_params)
    with start_run(f"xgboost_{run_id}", tags={"model": "xgboost", "run_id": run_id}):
        log_params(xgb_model.get_params())
        xgb_model = train_xgboost(xgb_model, X_train, y_train, X_test, y_test)
        preds = xgb_model.predict(X_test)
        metrics_dict = compute_metrics(inv(y_test), inv(preds), "XGBoost")
        log_metrics({k: v for k, v in metrics_dict.items() if k != "Model"})
        versioned_path = save_versioned_model(xgb_model, output_dir, "xgb_model", run_id)
        log_model_artifact(versioned_path)
        log.info(f"XGBoost: MAE={metrics_dict['MAE']:.4f}  R2={metrics_dict['R2']:.4f}")

    # ── LightGBM ─────────────────────────────────────────────────────────
    lgb_model = build_lightgbm(lgb_params)
    with start_run(f"lightgbm_{run_id}", tags={"model": "lightgbm", "run_id": run_id}):
        log_params(lgb_model.get_params())
        lgb_model = train_lightgbm(lgb_model, X_train, y_train, X_test, y_test)
        preds = lgb_model.predict(X_test)
        metrics_dict = compute_metrics(inv(y_test), inv(preds), "LightGBM")
        log_metrics({k: v for k, v in metrics_dict.items() if k != "Model"})
        versioned_path = save_versioned_model(lgb_model, output_dir, "lgb_model", run_id)
        log_model_artifact(versioned_path)
        log.info(f"LightGBM: MAE={metrics_dict['MAE']:.4f}  R2={metrics_dict['R2']:.4f}")

    # ── Run manifest ────────────────────────────────────────────────────────
    write_run_manifest(output_dir, run_id, {
        "xgb_params": xgb_model.get_params(),
        "lgb_params": lgb_model.get_params(),
        "metrics_xgb": compute_metrics(inv(y_test), inv(xgb_model.predict(X_test)), "XGBoost"),
        "metrics_lgb": compute_metrics(inv(y_test), inv(lgb_model.predict(X_test)), "LightGBM"),
    })

    return xgb_model, lgb_model, run_id
