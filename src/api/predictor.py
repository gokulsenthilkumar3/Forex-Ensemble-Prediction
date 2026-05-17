"""
src/api/predictor.py
=====================
Stateful singleton that loads all model artifacts once at startup
and exposes a clean predict() interface to the route handlers.

Fixed bugs (see review comment on PR #7):
  - Bug 1: predict() return type annotation corrected to Tuple[...,...,pd.DataFrame].
  - Bug 2: _preprocess() now preserves '_currency_code_orig' before one-hot encoding
           so that app.py can reconstruct correct currency codes for every row,
           including the currency that get_dummies(drop_first=True) removes.
  - Bug 3: Keras DL models are now loaded ONCE in load_artifacts() and cached
           in self._dl_models; _mc_uncertainty() reads from the cache, not disk.
  - Minor: remove_outliers_iqr is skipped during inference to prevent silent
           row drops; a warning is logged instead.
"""

from __future__ import annotations
import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

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
    All heavy I/O happens once in load_artifacts(); predict() is read-only.
    """

    def __init__(self, model_dir: str, config_path: str):
        self.model_dir             = model_dir
        self.feat_cfg              = load_config(config_path)
        self.scaler_y              = None
        self.per_currency_scalers: Optional[Dict] = None
        self._models: Dict         = {}
        self._dl_models: List      = []   # FIX Bug 3: cache DL models at startup
        self._artifacts_loaded     = False

    # ── Startup ────────────────────────────────────────────────────────────────

    def load_artifacts(self) -> None:
        """Load scalers, tree/stacking models, and DL Keras models from model_dir."""
        sy_path = os.path.join(self.model_dir, "scaler_y.pkl")
        pc_path = os.path.join(self.model_dir, "per_currency_scalers.pkl")

        if not os.path.exists(sy_path):
            raise FileNotFoundError(
                f"scaler_y.pkl not found in {self.model_dir}. Run training first."
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

        # FIX Bug 3: Load all Keras DL models ONCE at startup, cache in self._dl_models.
        # This avoids per-request disk I/O (100ms-1s per model) when uncertainty is requested.
        keras_files = glob.glob(os.path.join(self.model_dir, "*.keras"))
        if keras_files:
            try:
                import tensorflow as tf
                for kf in keras_files:
                    try:
                        m = tf.keras.models.load_model(kf, compile=False)
                        self._dl_models.append(m)
                        log.info(f"  Loaded DL model: {os.path.basename(kf)}")
                    except Exception as e:
                        log.warning(f"  Could not load DL model {kf}: {e}")
            except ImportError:
                log.warning("TensorFlow not available; MC-Dropout uncertainty disabled.")
        else:
            log.info("No Keras DL models found in model_dir; uncertainty will be zero.")

        self._artifacts_loaded = True
        log.info(
            f"ForexPredictor ready. "
            f"Models: {list(self._models.keys())}  "
            f"DL models: {len(self._dl_models)}"
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

    # ── Core preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply feature engineering + per-currency scaling.
        Returns (X_flat, df_processed).

        Changes from training pipeline:
        - remove_outliers_iqr is SKIPPED (would silently drop inference rows).
        - '_currency_code_orig' column is added before one-hot encoding so the
          caller can always recover the true currency regardless of drop_first.
        - '_orig_idx' column tracks original row order through sort_values.
        """
        # FIX Minor: Do NOT call remove_outliers_iqr during inference.
        # It drops rows that fall outside training IQR bounds, which causes
        # the response to have fewer predictions than submitted rows.
        n_input = len(df)
        log.debug(f"_preprocess: input rows={n_input}")

        # Preserve original row order
        df = df.reset_index(drop=True)
        df["_orig_idx"] = df.index

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

        if len(df) < n_input:
            log.warning(
                f"_preprocess: {n_input - len(df)} rows dropped after feature "
                "engineering (NaN from lag/rolling features at window boundaries). "
                "These rows will have no corresponding prediction."
            )

        # Per-currency scaling
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

        # FIX Bug 2: Preserve original currency code BEFORE one-hot encoding.
        # get_dummies(drop_first=True) removes the first alphabetical currency;
        # all its rows would have 0 in every OHE column, making idxmax() wrong.
        df["_currency_code_orig"] = df["currency_code"]

        df = pd.get_dummies(df, columns=["currency_code"], drop_first=True)
        bool_cols = df.select_dtypes("bool").columns
        if len(bool_cols):
            df[bool_cols] = df[bool_cols].astype(int)

        # Restore original row order (sort_values above re-ordered rows)
        df = df.sort_values("_orig_idx").reset_index(drop=True)

        feature_cols = [
            c for c in df.columns
            if c not in (
                "date", "exchange_rate", "target",
                "_orig_idx", "_currency_code_orig",
            )
        ]
        X = df[feature_cols].values.astype(np.float32)
        return X, df

    # ── Public predict interface ───────────────────────────────────────────────

    def predict(
        self,
        rows: List[dict],
        model_name: str = "stacking",
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        """
        Run the full inference pipeline on a list of raw input dicts.

        Parameters
        ----------
        rows               : Raw input rows (dicts with 'date', 'currency_code', etc.)
        model_name         : Which artifact to use.
        return_uncertainty : If True, also returns MC-Dropout uncertainty (std dev).

        Returns
        -------
        FIX Bug 1: Corrected return type annotation from 2-tuple to 3-tuple.
        (predictions, uncertainties, df_proc)
          predictions   : np.ndarray (N,) — inverse-scaled exchange rates.
          uncertainties : np.ndarray (N,) or None.
          df_proc       : Processed DataFrame (for response reconstruction in app.py).
        """
        if not self._artifacts_loaded:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")
        if model_name not in self._models:
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Loaded: {list(self._models.keys())}"
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
            uncertainties = self._mc_uncertainty(X, n_samples=30)

        return preds, uncertainties, df_proc

    # ── MC-Dropout uncertainty (uses cached DL models) ──────────────────────────

    def _mc_uncertainty(
        self, X_flat: np.ndarray, n_samples: int = 30
    ) -> np.ndarray:
        """
        FIX Bug 3: Uses self._dl_models (loaded once at startup) instead of
        reloading Keras models from disk on every request.

        Returns aggregated std across DL models, or zeros if none loaded.
        """
        if not self._dl_models:
            log.warning("No DL models cached; uncertainty will be zero.")
            return np.zeros(len(X_flat), dtype=np.float32)

        try:
            from src.models.deep_learning import mc_predict
            all_stds = []
            for m in self._dl_models:
                try:
                    timesteps = m.input_shape[1]
                    if X_flat.shape[0] < timesteps:
                        continue
                    X_seq = np.stack(
                        [X_flat[i: i + timesteps]
                         for i in range(len(X_flat) - timesteps)],
                        axis=0,
                    )
                    _, std = mc_predict(m, X_seq, n_samples=n_samples)
                    padded = np.concatenate([np.zeros(timesteps), std])
                    all_stds.append(padded[: len(X_flat)])
                except Exception as e:
                    log.debug(f"MC-Dropout failed for a DL model: {e}")
            if all_stds:
                return np.mean(
                    np.stack(all_stds, axis=0), axis=0
                ).astype(np.float32)
        except ImportError as e:
            log.warning(f"MC-Dropout import error: {e}")

        return np.zeros(len(X_flat), dtype=np.float32)
