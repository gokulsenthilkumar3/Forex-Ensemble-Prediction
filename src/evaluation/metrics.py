"""
src/evaluation/metrics.py
==========================
Regression and directional accuracy metrics for Forex prediction.
"""

from __future__ import annotations
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Compute a standard set of regression + directional accuracy metrics.

    Parameters
    ----------
    y_true : Ground-truth exchange rates (1-D array).
    y_pred : Predicted exchange rates (1-D array).
    name   : Model display name for the returned dict.

    Returns
    -------
    dict with keys: Model, MSE, MAE, RMSE, MAPE, R2, DA.

    Notes
    -----
    DA (Directional Accuracy) measures the fraction of consecutive
    time steps where the sign of the change is predicted correctly.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mse  = float(np.mean((y_true - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))

    nonzero = y_true != 0
    mape = float(
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    ) if nonzero.any() else float("nan")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    da = float(
        np.mean(
            np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
        ) * 100
    ) if len(y_true) > 1 else 0.0

    return {
        "Model": name,
        "MSE":   mse,
        "MAE":   mae,
        "RMSE":  rmse,
        "MAPE":  mape,
        "R2":    r2,
        "DA":    da,
    }
