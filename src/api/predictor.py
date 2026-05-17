"""
src/api/predictor.py
=====================
Stateful singleton that loads all model artifacts once at startup
and exposes a clean predict() interface to the route handlers.

Design principles
-----------------
- All heavy I/O (model loading, scaler loading) happens ONCE at startup
  inside `load_artifacts()`, not per request.
- `predict()` is thread-safe because it only reads shared state.
- Uncertainty estimation via MC-Dropout is opt-in per request.
"""

from __future__ import annotations
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.data.cleaner import remove_outliers_iqr
from src.features.engineer import add_features, load_config
from src.features.cross_currency import add_cross_currency_features

log = logging.getLogger(__name__)

_MODEL_FILE_MAP = {
    "lgb":      "lgb_model.pkl",
    "xgb":      "xgb_model.pkl",
    "stacking": "stacking_meta.pkl",
}


class ForexPredictor:
    """
    Stateful predictor singleton.

    Attributes
    ----------
    model_dir            : Path to the artifact directory.
    feat_cfg             : Loaded features.yaml config.
    scaler_y             : Target inverse-transform scaler.
    per_currency_scalers : Dict of per-currency RobustScalers.
    _models              : Lazy-loaded dict of model name -> fitted model.
    """

    def __init__(self, model_dir: str, config_path: str):
        self.model_dir             = model_dir
        self.feat_cfg              = load_config(config_path)
        self.scaler_y              = None
        self.per_currency_scalers  = None
        self._models: Dict         = {}
        self._artifacts_loaded     = False

    # ── Startup ────────────────────────────────────────────────────────────────

    def load_artifacts(self) -> None:
        """Load scalers and all available models from model_dir."""
        sy_path = os.path.join(self.model_dir, "scaler_y.pkl")
        pc_path = os.path.join(self.model_dir, "per_currency_scalers.pkl")

        if not os.path.exists(sy_path):
            raise FileNotFoundError(
                f"scaler_y.pkl not found in {self.model_dir}. "
                "Run training first."
            )
        self.scaler_y             = joblib.load(sy_path)
        self.per_currency_scalers = joblib.load(pc_path)
        log.info(f"Scalers loaded from {self.model_dir}")

        for name, fname in _MODEL_FILE_MAP.items():
            fpath = os.path.join(self.model_dir, fname)
            if os.path.exists(fpath):
                self._models[name] = joblib.load(fpath)
                log.info(f"  Loaded model: {name} ({fname})")
            else:
                log.warning(f"  Model artifact not found, skipping: {fname}")

        self._artifacts_loaded = True
        log.info(
            f"ForexPredictor ready. Available models: {list(self._models.keys())}"
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def artifacts_loaded(self) -> bool:
        return self._artifacts_loaded

    @property
    def available_models(self) -> List[str]:
        return list(self._models.keys())

    @property
    def n_currencies(self) -> int:
        return len(self.per_currency_scalers) if self.per_currency_scalers else 0

    @property
    def artifact_files(self) -> List[str]:
        if not os.path.isdir(self.model_dir):
            return []
        return [
            f for f in os.listdir(self.model_dir)
            if f.endswith((".pkl", ".keras", ".json"))
        ]

    # ── Core preprocessing (reuses training pipeline) ─────────────────────────

    def _preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply the same feature engineering + per-currency scaling pipeline
        used during training.

        Returns (X_flat, df_processed).
        """
        df = remove_outliers_iqr(df)

        dfs = []
        for code, grp in df.groupby("currency_code"):
            g = add_features(grp, self.feat_cfg)
            g["currency_code"] = code
            dfs.append(g)
        df = (
            pd.concat(dfs, ignore_index=True)
            .sort_values(["currency_code", "date"])
            .reset_index(drop=True)
        )
        df = add_cross_currency_features(df, self.feat_cfg["cross_currency"])
        df = df.dropna().reset_index(drop=True)

        scaled = []
        for code, grp in df.groupby("currency_code"):
            grp = grp.copy()
            if code in self.per_currency_scalers:
                sc   = self.per_currency_scalers[code]["scaler"]
                cols = [
                    c for c in self.per_currency_scalers[code]["cols"]
                    if c in grp.columns
                ]
                grp[cols] = sc.transform(grp[cols].values)
            else:
                log.warning(
                    f"No scaler found for currency '{code}'. Using raw features."
                )
            scaled.append(grp)
        df = pd.concat(scaled, ignore_index=True)

        df = pd.get_dummies(df, columns=["currency_code"], drop_first=True)
        bool_cols = df.select_dtypes("bool").columns
        if len(bool_cols):
            df[bool_cols] = df[bool_cols].astype(int)

        feature_cols = [
            c for c in df.columns
            if c not in ("date", "exchange_rate", "target")
        ]
        X = df[feature_cols].values.astype(np.float32)
        return X, df

    # ── Public predict interface ───────────────────────────────────────────────

    def predict(
        self,
        rows: List[dict],
        model_name: str = "stacking",
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run the full inference pipeline on a list of raw input dicts.

        Parameters
        ----------
        rows               : Raw input rows (dicts with 'date', 'currency_code', etc.)
        model_name         : Which artifact to use.
        return_uncertainty : If True, also returns MC-Dropout uncertainty
                             array (requires Dropout layers in loaded DL models).

        Returns
        -------
        (predictions, uncertainties)
        predictions   : np.ndarray of shape (N,) — inverse-scaled exchange rates.
        uncertainties : np.ndarray of shape (N,) or None.
        """
        if not self._artifacts_loaded:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")
        if model_name not in self._models:
            available = list(self._models.keys())
            raise ValueError(
                f"Model '{model_name}' not available. Loaded: {available}"
            )

        df_raw = pd.DataFrame(rows)
        df_raw["date"] = pd.to_datetime(df_raw["date"], dayfirst=True, format="mixed")
        if "currency" in df_raw.columns:
            df_raw = df_raw.drop(columns=["currency"])

        X, df_proc = self._preprocess(df_raw)
        log.info(f"Inference input shape: {X.shape}  model={model_name}")

        model        = self._models[model_name]
        preds_scaled = np.asarray(model.predict(X)).reshape(-1, 1)
        preds        = self.scaler_y.inverse_transform(preds_scaled).flatten()

        uncertainties = None
        if return_uncertainty:
            # Try to load DL sub-models for MC-Dropout
            uncertainties = self._mc_uncertainty(X, n_samples=30)

        return preds, uncertainties, df_proc

    def _mc_uncertainty(
        self, X_flat: np.ndarray, n_samples: int = 30
    ) -> Optional[np.ndarray]:
        """
        Attempt MC-Dropout uncertainty estimation using any available
        Keras DL models in the model_dir.
        Returns aggregated std across models, or zeros if none available.
        """
        import glob
        keras_files = glob.glob(os.path.join(self.model_dir, "*.keras"))
        if not keras_files:
            log.warning("No Keras models found; uncertainty will be zero.")
            return np.zeros(len(X_flat), dtype=np.float32)

        try:
            import tensorflow as tf
            all_stds = []
            for kf in keras_files:
                try:
                    m = tf.keras.models.load_model(kf, compile=False)
                    # Build sequences (single-step window fallback)
                    timesteps = m.input_shape[1]
                    if X_flat.shape[0] < timesteps:
                        continue
                    X_seq = np.stack(
                        [X_flat[i:i + timesteps]
                         for i in range(len(X_flat) - timesteps)],
                        axis=0,
                    )
                    from src.models.deep_learning import mc_predict
                    _, std = mc_predict(m, X_seq, n_samples=n_samples)
                    # Pad to original length
                    padded = np.concatenate([np.zeros(timesteps), std])
                    all_stds.append(padded[: len(X_flat)])
                except Exception as e:
                    log.debug(f"MC-Dropout failed for {kf}: {e}")
            if all_stds:
                return np.mean(np.stack(all_stds, axis=0), axis=0).astype(np.float32)
        except ImportError:
            log.warning("TensorFlow not available; uncertainty will be zero.")

        return np.zeros(len(X_flat), dtype=np.float32)
