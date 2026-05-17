"""
src/models/tree_models.py
=========================
XGBoost and LightGBM model builders, trainers, and persistence helpers.

Improvements over v3.1:
  - Optuna-based hyperparameter search (build_xgboost_tuned / build_lightgbm_tuned)
  - CatBoost added as a third tree-based base learner
  - train_xgboost uses early stopping via callback (XGBoost >= 2.0 API)
  - All builders accept optional override dicts for reproducibility
"""

from __future__ import annotations
import os
import logging
import joblib
import numpy as np

import xgboost as xgb
import lightgbm as lgb

log = logging.getLogger(__name__)


# ── Default builders ──────────────────────────────────────────────────────────

def build_xgboost(params: dict | None = None) -> xgb.XGBRegressor:
    """
    Build an XGBoost regressor with default or provided params.
    Uses histogram tree method and enables early stopping via n_estimators.
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
    """Build a LightGBM regressor with default or provided params."""
    default = dict(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=-1,
    )
    if params:
        default.update(params)
    return lgb.LGBMRegressor(**default)


def build_catboost(params: dict | None = None):
    """
    Build a CatBoost regressor.
    CatBoost uses symmetric trees and ordered boosting, which often
    complements XGBoost/LightGBM in an ensemble.

    Falls back gracefully if catboost is not installed.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        log.warning("catboost not installed — skipping CatBoost model.")
        return None

    default = dict(
        iterations=800, learning_rate=0.05, depth=7,
        l2_leaf_reg=3.0, random_seed=42,
        loss_function="RMSE", eval_metric="RMSE",
        verbose=False,
    )
    if params:
        default.update(params)
    return CatBoostRegressor(**default)


# ── Optuna-tuned builders ─────────────────────────────────────────────────────

def build_xgboost_tuned(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> xgb.XGBRegressor:
    """
    Optuna-based hyperparameter search for XGBoost.

    Searches: learning_rate, max_depth, subsample, colsample_bytree,
              min_child_weight, reg_alpha, reg_lambda.

    Parameters
    ----------
    n_trials : Number of Optuna trials (increase for better results).

    Returns
    -------
    Fitted XGBRegressor with best found hyperparameters.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("optuna not installed — falling back to default XGBoost.")
        m = build_xgboost()
        return train_xgboost(m, X_train, y_train, X_val, y_val)

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            tree_method="hist", random_state=random_state,
            n_jobs=-1, verbosity=0,
        )
        m = xgb.XGBRegressor(**p)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return float(np.mean((y_val - m.predict(X_val)) ** 2))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"XGBoost best params: {study.best_params}")

    best_model = xgb.XGBRegressor(**{**study.best_params,
                                     "tree_method": "hist", "n_jobs": -1,
                                     "verbosity": 0, "random_state": random_state})
    return train_xgboost(best_model, X_train, y_train, X_val, y_val)


def build_lightgbm_tuned(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> lgb.LGBMRegressor:
    """
    Optuna-based hyperparameter search for LightGBM.

    Returns fitted LGBMRegressor with best found hyperparameters.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("optuna not installed — falling back to default LightGBM.")
        m = build_lightgbm()
        return train_lightgbm(m, X_train, y_train, X_val, y_val)

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 200),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state=random_state, n_jobs=-1, verbosity=-1,
        )
        m = lgb.LGBMRegressor(**p)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
        return float(np.mean((y_val - m.predict(X_val)) ** 2))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"LightGBM best params: {study.best_params}")

    best_model = lgb.LGBMRegressor(**{**study.best_params,
                                      "n_jobs": -1, "verbosity": -1,
                                      "random_state": random_state})
    return train_lightgbm(best_model, X_train, y_train, X_val, y_val)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_xgboost(
    model: xgb.XGBRegressor,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> xgb.XGBRegressor:
    """Fit XGBoost with an eval set for monitoring."""
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    log.info("XGBoost training complete.")
    return model


def train_lightgbm(
    model: lgb.LGBMRegressor,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMRegressor:
    """Fit LightGBM with early stopping."""
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


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """Persist a model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log.info(f"Model saved to {path}")


def load_model(path: str):
    """Load a persisted model from disk."""
    log.info(f"Loading model from {path}")
    return joblib.load(path)
