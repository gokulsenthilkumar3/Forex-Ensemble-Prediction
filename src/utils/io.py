"""
src/utils/io.py
================
I/O helpers: sequence builder, scaler persistence, results export.
"""

from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd
import joblib

log = logging.getLogger(__name__)


def make_sequences(X: np.ndarray, y: np.ndarray, timesteps: int):
    """
    Create sliding-window sequences for time-series deep learning models.

    Parameters
    ----------
    X         : 2-D feature array (n_samples, n_features).
    y         : 1-D or 2-D target array (n_samples,) or (n_samples, 1).
    timesteps : Lookback window length.

    Returns
    -------
    (X_seq, y_seq) where X_seq.shape = (n, timesteps, n_features)
    and y_seq.shape = (n,).
    """
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i: i + timesteps])
        ys.append(y[i + timesteps])
    return np.array(Xs), np.array(ys)


def save_results(results: list[dict], output_dir: str, filename: str = "model_comparison.csv") -> None:
    """
    Save a list of metrics dicts to a CSV file.

    Parameters
    ----------
    results    : List of dicts from compute_metrics().
    output_dir : Target directory.
    filename   : Output CSV filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    pd.DataFrame(results).to_csv(path, index=False)
    log.info(f"Results saved to {path}")


def save_scaler(scaler, output_dir: str, name: str) -> None:
    """Persist a fitted sklearn scaler using joblib."""
    path = os.path.join(output_dir, f"{name}.pkl")
    joblib.dump(scaler, path)
    log.info(f"Scaler '{name}' saved to {path}")


def load_scaler(output_dir: str, name: str):
    """Load a persisted sklearn scaler from disk."""
    path = os.path.join(output_dir, f"{name}.pkl")
    log.info(f"Loading scaler '{name}' from {path}")
    return joblib.load(path)
