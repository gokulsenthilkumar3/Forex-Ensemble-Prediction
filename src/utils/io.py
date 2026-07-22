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


class PerCurrencyTargetScaler:
    """Scales target values independently for each currency using MinMaxScaler."""
    def __init__(self, feature_range=(0.05, 0.95)):
        self.feature_range = feature_range
        self.scalers = {}

    def fit(self, y: np.ndarray, currencies: np.ndarray) -> PerCurrencyTargetScaler:
        from sklearn.preprocessing import MinMaxScaler
        y_flat = np.asarray(y).flatten()
        currencies_flat = np.asarray(currencies).flatten()
        df_temp = pd.DataFrame({"y": y_flat, "curr": currencies_flat})
        for curr, grp in df_temp.groupby("curr"):
            sc = MinMaxScaler(feature_range=self.feature_range)
            sc.fit(grp["y"].values.reshape(-1, 1))
            self.scalers[curr] = sc
        return self

    def transform(self, y: np.ndarray, currencies: np.ndarray) -> np.ndarray:
        y_flat = np.asarray(y).flatten()
        currencies_flat = np.asarray(currencies).flatten()
        y_scaled = np.zeros_like(y_flat)
        df_temp = pd.DataFrame({"y": y_flat, "curr": currencies_flat})
        for curr, grp in df_temp.groupby("curr"):
            if curr in self.scalers:
                sc = self.scalers[curr]
                y_scaled[grp.index] = sc.transform(grp["y"].values.reshape(-1, 1)).flatten()
            else:
                y_scaled[grp.index] = grp["y"].values
        return y_scaled.reshape(-1, 1)

    def fit_transform(self, y: np.ndarray, currencies: np.ndarray) -> np.ndarray:
        self.fit(y, currencies)
        return self.transform(y, currencies)

    def inverse_transform(self, y: np.ndarray, currencies: np.ndarray | None = None) -> np.ndarray:
        y_flat = np.asarray(y).flatten()
        if currencies is None:
            # Fallback to returning raw array if no currencies provided
            return y_flat.reshape(-1, 1)
        currencies_flat = np.asarray(currencies).flatten()
        y_inv = np.zeros_like(y_flat)
        df_temp = pd.DataFrame({"y": y_flat, "curr": currencies_flat})
        for curr, grp in df_temp.groupby("curr"):
            if curr in self.scalers:
                sc = self.scalers[curr]
                y_inv[grp.index] = sc.inverse_transform(grp["y"].values.reshape(-1, 1)).flatten()
            else:
                y_inv[grp.index] = grp["y"].values
        return y_inv.reshape(-1, 1)

