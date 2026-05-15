"""
src/models/tree_models.py
=========================
XGBoost and LightGBM model builders, trainers, and persistence helpers.
"""

from __future__ import annotations
import os
import logging
import joblib
import numpy as np

import xgboost as xgb
import lightgbm as lgb

log = logging.getLogger(__name__)


def build_xgboost(params: dict | None = None) -> xgb.XGBRegressor:
    """
    Build an XGBoost regressor with default or provided params.

    Parameters
    ----------
    params : Optional dict of XGBRegressor keyword arguments.
             Defaults to a reasonable baseline if None.

    Returns
    -------
    Unfitted xgb.XGBRegressor instance.
    """
    default = dict(
        n_estimators=800, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        tree_method="hist", n_jobs=-1, verbosity=0,
    )
    if params:
        default.update(params)
    return xgb.XGBRegressor(**default)


def build_lightgbm(params: dict | None = None) -> lgb.LGBMRegressor:
    """
    Build a LightGBM regressor with default or provided params.

    Parameters
    ----------
    params : Optional dict of LGBMRegressor keyword arguments.

    Returns
    -------
    Unfitted lgb.LGBMRegressor instance.
    """
    default = dict(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=-1,
    )
    if params:
        default.update(params)
    return lgb.LGBMRegressor(**default)


def train_xgboost(
    model: xgb.XGBRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBRegressor:
    """
    Fit XGBoost with an eval set for early stopping monitoring.

    Returns the fitted model.
    """
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    log.info("XGBoost training complete.")
    return model


def train_lightgbm(
    model: lgb.LGBMRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMRegressor:
    """
    Fit LightGBM with early stopping.

    Returns the fitted model.
    """
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    log.info("LightGBM training complete.")
    return model


def save_model(model, path: str) -> None:
    """Persist a model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log.info(f"Model saved to {path}")


def load_model(path: str):
    """Load a persisted model from disk."""
    log.info(f"Loading model from {path}")
    return joblib.load(path)
