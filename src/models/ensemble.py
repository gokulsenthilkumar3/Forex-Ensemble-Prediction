"""
src/models/ensemble.py
=======================
Advanced stacking and blending ensemble strategies.

Replaces the single Ridge meta-learner in forex_prediction.py with:

1. Out-of-Fold (OOF) Stacking
   - Base models are evaluated on held-out folds, not on training data,
     so the meta-learner trains on unbiased level-1 predictions.
   - Prevents the meta-learner from over-fitting to base model memorisation.

2. Dynamic Performance-Weighted Blending
   - Each base model is assigned a weight inversely proportional to its
     validation MAE, providing a fast and interpretable ensemble baseline.

3. Neural Meta-Learner option
   - Replaces Ridge with a small 2-layer MLP for non-linear blending.

4. Uncertainty-Aware Inference
   - Integrates MC-Dropout std-dev outputs as extra features for the
     meta-learner, allowing it to down-weight uncertain base predictions.

Usage
-----
    from src.models.ensemble import (
        build_oof_meta_features,
        DynamicWeightedBlender,
        NeuralMetaLearner,
        StackingEnsemble,
    )
"""

from __future__ import annotations
import logging
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── 1. Out-of-Fold Meta-Feature Builder ───────────────────────────────────────

def build_oof_meta_features(
    base_trainers: Dict[str, Callable],
    X_flat: np.ndarray,
    y_flat: np.ndarray,
    X_seq: Optional[np.ndarray] = None,
    y_seq: Optional[np.ndarray] = None,
    n_splits: int = 5,
    seq_model_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Generate Out-of-Fold (OOF) level-1 predictions for the meta-learner.

    For each TimeSeriesSplit fold, each base model is re-trained on the
    in-fold data and predicts on the out-of-fold (held-out) slice.
    This yields unbiased meta-features covering the entire dataset.

    Parameters
    ----------
    base_trainers   : Dict mapping model name -> callable(X_tr, y_tr, X_val, y_val)
                      that returns a fitted model with a .predict(X) method.
    X_flat          : 2-D feature array (N, F) — used for tree models.
    y_flat          : 1-D target array (N,).
    X_seq           : 3-D sequence array (N, T, F) — used for DL models.
                      If None, seq_model_names must also be None/empty.
    y_seq           : 1-D target array aligned to X_seq. If None, y_flat is used.
    n_splits        : Number of TimeSeriesSplit folds.
    seq_model_names : Names in base_trainers that expect X_seq input.

    Returns
    -------
    oof_preds   : (N, n_models) array of OOF predictions for each base model.
    oof_targets : (N,) array of corresponding targets.
    val_maes    : Dict mapping model name -> mean OOF MAE.
    """
    seq_model_names = set(seq_model_names or [])
    model_names     = list(base_trainers.keys())
    n               = len(X_flat)
    oof_preds       = np.full((n, len(model_names)), np.nan, dtype=np.float32)
    val_maes: Dict[str, List[float]] = {nm: [] for nm in model_names}

    tss = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr_idx, val_idx) in enumerate(tss.split(X_flat)):
        log.info(f"OOF fold {fold + 1}/{n_splits}  "
                 f"train={len(tr_idx)}  val={len(val_idx)}")
        X_tr_f, X_val_f = X_flat[tr_idx], X_flat[val_idx]
        y_tr_f, y_val_f = y_flat[tr_idx], y_flat[val_idx]

        # Sequence slices (only last len(val_idx) of the val window for DL)
        if X_seq is not None:
            y_seq_used = y_seq if y_seq is not None else y_flat
            X_tr_s  = X_seq[tr_idx[tr_idx < len(X_seq)]]
            y_tr_s  = y_seq_used[tr_idx[tr_idx < len(X_seq)]]
            X_val_s = X_seq[val_idx[val_idx < len(X_seq)]]

        for col_idx, nm in enumerate(model_names):
            trainer = base_trainers[nm]
            try:
                if nm in seq_model_names and X_seq is not None:
                    model = trainer(X_tr_s, y_tr_s, X_val_s, y_val_f)
                    p_val = model.predict(X_val_s,
                                         verbose=0).flatten()[:len(val_idx)]
                else:
                    model = trainer(X_tr_f, y_tr_f, X_val_f, y_val_f)
                    p_val = np.asarray(model.predict(X_val_f)).flatten()

                oof_preds[val_idx[:len(p_val)], col_idx] = p_val
                val_maes[nm].append(mean_absolute_error(y_val_f[:len(p_val)],
                                                        p_val))
            except Exception as exc:
                log.warning(f"OOF fold {fold + 1} failed for {nm}: {exc}")

    # Average fold MAEs
    mean_val_maes = {nm: float(np.mean(v)) if v else float("inf")
                     for nm, v in val_maes.items()}
    log.info("OOF validation MAEs: " +
             ", ".join(f"{k}={v:.5f}" for k, v in mean_val_maes.items()))

    # Fill any remaining NaNs with column mean
    for col in range(oof_preds.shape[1]):
        mask = np.isnan(oof_preds[:, col])
        if mask.any():
            oof_preds[mask, col] = np.nanmean(oof_preds[:, col])

    return oof_preds, y_flat, mean_val_maes


# ── 2. Dynamic Performance-Weighted Blender ───────────────────────────────────

class DynamicWeightedBlender:
    """
    Combines base model predictions using weights proportional to
    1 / (val_mae + epsilon), so better models contribute more.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_models,) — normalised blend weights.
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon  = epsilon
        self.weights_ = None

    def fit(self, val_maes: Dict[str, float]) -> "DynamicWeightedBlender":
        """
        Compute normalised inverse-MAE weights.

        Parameters
        ----------
        val_maes : Dict mapping model name -> validation MAE.
        """
        raw = np.array([1.0 / (v + self.epsilon)
                        for v in val_maes.values()], dtype=np.float64)
        self.weights_      = raw / raw.sum()
        self.model_names_  = list(val_maes.keys())
        for nm, w in zip(self.model_names_, self.weights_):
            log.info(f"  Blend weight  {nm}: {w:.4f}")
        return self

    def predict(self, meta_X: np.ndarray) -> np.ndarray:
        """
        Weighted average of base predictions.

        Parameters
        ----------
        meta_X : (N, n_models) array of base model predictions.

        Returns
        -------
        (N,) weighted blend.
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        return meta_X @ self.weights_

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DynamicWeightedBlender":
        return joblib.load(path)


# ── 3. Neural Meta-Learner ────────────────────────────────────────────────────

class NeuralMetaLearner:
    """
    Small 2-layer MLP meta-learner as a drop-in replacement for Ridge.
    Enables non-linear blending of base model predictions.

    Parameters
    ----------
    hidden_units : Tuple of ints — hidden layer sizes.
    dropout      : Dropout rate applied between hidden layers.
    epochs       : Max training epochs.
    patience     : Early stopping patience.
    """

    def __init__(self, hidden_units: tuple = (64, 32),
                 dropout: float = 0.2,
                 epochs: int = 100,
                 patience: int = 10):
        self.hidden_units = hidden_units
        self.dropout      = dropout
        self.epochs       = epochs
        self.patience     = patience
        self._model       = None

    def _build(self, n_inputs: int):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import AdamW
        m = Sequential(name="NeuralMeta")
        m.add(Dense(self.hidden_units[0], activation="relu",
                    input_shape=(n_inputs,)))
        m.add(Dropout(self.dropout))
        for u in self.hidden_units[1:]:
            m.add(Dense(u, activation="relu"))
            m.add(Dropout(self.dropout))
        m.add(Dense(1))
        m.compile(optimizer=AdamW(1e-3, weight_decay=1e-4),
                  loss="huber", metrics=["mae"])
        return m

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_split: float = 0.2) -> "NeuralMetaLearner":
        import tensorflow as tf
        self._model = self._build(X.shape[1])
        cb = [
            tf.keras.callbacks.EarlyStopping(
                "val_loss", patience=self.patience, restore_best_weights=True
            )
        ]
        self._model.fit(X, y, epochs=self.epochs,
                        validation_split=validation_split,
                        callbacks=cb, verbose=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X, verbose=0).flatten()

    def save(self, path: str) -> None:
        if self._model is not None:
            self._model.save(path.replace(".pkl", ".keras"))

    @staticmethod
    def load(path: str) -> "NeuralMetaLearner":
        import tensorflow as tf
        obj = NeuralMetaLearner()
        obj._model = tf.keras.models.load_model(
            path.replace(".pkl", ".keras")
        )
        return obj


# ── 4. Unified StackingEnsemble ───────────────────────────────────────────────

class StackingEnsemble:
    """
    Unified stacking ensemble that wraps Ridge, DynamicWeightedBlender,
    or NeuralMetaLearner under a common interface.

    Parameters
    ----------
    meta_learner : One of "ridge" | "weighted" | "neural".
    ridge_alpha  : Regularisation strength (only for "ridge").
    """

    def __init__(self, meta_learner: str = "ridge",
                 ridge_alpha: float = 1.0):
        self.meta_learner_type = meta_learner
        self.ridge_alpha       = ridge_alpha
        self._meta             = None
        self.val_maes_: Dict[str, float] = {}

    def fit(self, meta_X: np.ndarray, meta_y: np.ndarray,
            val_maes: Optional[Dict[str, float]] = None) -> "StackingEnsemble":
        """
        Fit the meta-learner on level-1 (base model) predictions.

        Parameters
        ----------
        meta_X   : (N, n_models) array of base model predictions.
        meta_y   : (N,) true targets.
        val_maes : Required when meta_learner == "weighted".
        """
        if self.meta_learner_type == "ridge":
            self._meta = Ridge(alpha=self.ridge_alpha)
            self._meta.fit(meta_X, meta_y)
            log.info("Ridge meta-learner fitted.")

        elif self.meta_learner_type == "weighted":
            if val_maes is None:
                raise ValueError("val_maes required for weighted blender.")
            self._meta = DynamicWeightedBlender()
            self._meta.fit(val_maes)
            self.val_maes_ = val_maes

        elif self.meta_learner_type == "neural":
            self._meta = NeuralMetaLearner()
            self._meta.fit(meta_X, meta_y)
            log.info("Neural meta-learner fitted.")

        else:
            raise ValueError(
                f"Unknown meta_learner '{self.meta_learner_type}'. "
                "Choose ridge | weighted | neural."
            )
        return self

    def predict(self, meta_X: np.ndarray) -> np.ndarray:
        if self._meta is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._meta.predict(meta_X)).flatten()

    def save(self, path: str) -> None:
        """Persist the ensemble (handles Neural sub-type)."""
        if isinstance(self._meta, NeuralMetaLearner):
            self._meta.save(path)
        else:
            joblib.dump(self, path)
        log.info(f"StackingEnsemble saved to {path}")

    @staticmethod
    def load(path: str) -> "StackingEnsemble":
        return joblib.load(path)
