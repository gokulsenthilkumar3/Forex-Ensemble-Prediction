"""
src/models/ensemble.py
=======================
Advanced stacking and blending ensemble strategies.

Fixed bugs (see review comment on PR #6):
  - Bug 1: OOF sequence index misalignment corrected via explicit offset.
  - Bug 2: NeuralMetaLearner.save/load now persists wrapper metadata + Keras weights.
           StackingEnsemble.load handles neural variant correctly.
  - Bug 3: DynamicWeightedBlender.predict uses np.atleast_2d to handle single-row input.
  - Minor: Removed unused BatchNormalization import; removed redundant tf import in fit().
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
    X_seq           : 3-D sequence array (N_seq, T, F) — used for DL models.
                      N_seq = N - TIMESTEPS. If None, seq_model_names must be empty.
    y_seq           : 1-D target array of length N_seq aligned to X_seq.
                      If None, y_flat[-N_seq:] is used.
    n_splits        : Number of TimeSeriesSplit folds.
    seq_model_names : Names in base_trainers that expect X_seq input.

    Returns
    -------
    oof_preds   : (N, n_models) array of OOF predictions.
    oof_targets : (N,) array — same as y_flat.
    val_maes    : Dict mapping model name -> mean OOF MAE.
    """
    seq_model_names = set(seq_model_names or [])
    model_names     = list(base_trainers.keys())
    n               = len(X_flat)
    oof_preds       = np.full((n, len(model_names)), np.nan, dtype=np.float32)
    val_maes: Dict[str, List[float]] = {nm: [] for nm in model_names}

    # FIX (Bug 1): X_seq is shorter than X_flat by TIMESTEPS rows.
    # offset is the number of flat rows that have NO corresponding sequence row.
    # Flat index i maps to sequence index (i - offset), valid only when i >= offset.
    offset = (len(X_flat) - len(X_seq)) if X_seq is not None else 0
    y_seq_used = y_seq if (y_seq is not None) else (y_flat[offset:] if X_seq is not None else None)

    tss = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr_idx, val_idx) in enumerate(tss.split(X_flat)):
        log.info(
            f"OOF fold {fold + 1}/{n_splits}  "
            f"train={len(tr_idx)}  val={len(val_idx)}"
        )
        X_tr_f, X_val_f = X_flat[tr_idx], X_flat[val_idx]
        y_tr_f, y_val_f = y_flat[tr_idx], y_flat[val_idx]

        # Build sequence sub-arrays using the explicit offset.
        # Only indices >= offset have a valid sequence row.
        if X_seq is not None:
            tr_seq_mask  = tr_idx >= offset
            val_seq_mask = val_idx >= offset

            tr_seq_idx  = tr_idx[tr_seq_mask]  - offset   # indices into X_seq
            val_seq_idx = val_idx[val_seq_mask] - offset

            X_tr_s  = X_seq[tr_seq_idx]
            y_tr_s  = y_seq_used[tr_seq_idx]
            X_val_s = X_seq[val_seq_idx]
            # flat val indices that have a valid sequence row
            val_flat_seq = val_idx[val_seq_mask]

        for col_idx, nm in enumerate(model_names):
            trainer = base_trainers[nm]
            try:
                if nm in seq_model_names and X_seq is not None:
                    model  = trainer(X_tr_s, y_tr_s, X_val_s, y_seq_used[val_seq_idx])
                    p_val  = np.asarray(
                        model.predict(X_val_s, verbose=0)
                    ).flatten()
                    # Write predictions back to the CORRECT flat positions
                    write_idx = val_flat_seq[:len(p_val)]
                    oof_preds[write_idx, col_idx] = p_val[:len(write_idx)]
                    val_maes[nm].append(
                        mean_absolute_error(
                            y_seq_used[val_seq_idx[:len(p_val)]],
                            p_val[:len(val_seq_idx)],
                        )
                    )
                else:
                    model = trainer(X_tr_f, y_tr_f, X_val_f, y_val_f)
                    p_val = np.asarray(model.predict(X_val_f)).flatten()
                    oof_preds[val_idx[:len(p_val)], col_idx] = p_val
                    val_maes[nm].append(
                        mean_absolute_error(y_val_f[:len(p_val)], p_val)
                    )
            except Exception as exc:
                log.warning(f"OOF fold {fold + 1} failed for {nm}: {exc}")

    mean_val_maes = {
        nm: float(np.mean(v)) if v else float("inf")
        for nm, v in val_maes.items()
    }
    log.info(
        "OOF validation MAEs: "
        + ", ".join(f"{k}={v:.5f}" for k, v in mean_val_maes.items())
    )

    # Fill remaining NaNs (folds that failed) with column mean
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
        self.epsilon      = epsilon
        self.weights_     = None
        self.model_names_ = None

    def fit(self, val_maes: Dict[str, float]) -> "DynamicWeightedBlender":
        raw = np.array(
            [1.0 / (v + self.epsilon) for v in val_maes.values()],
            dtype=np.float64,
        )
        self.weights_     = raw / raw.sum()
        self.model_names_ = list(val_maes.keys())
        for nm, w in zip(self.model_names_, self.weights_):
            log.info(f"  Blend weight  {nm}: {w:.4f}")
        return self

    def predict(self, meta_X: np.ndarray) -> np.ndarray:
        """
        Weighted average of base predictions.

        FIX (Bug 3): np.atleast_2d ensures a single-row input of shape
        (n_models,) is promoted to (1, n_models) before the dot product,
        preventing the result from collapsing to a scalar.
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.atleast_2d(meta_X) @ self.weights_   # always returns (N,) or (1,)

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

    FIX (Bug 2): save() now persists both the Keras model weights AND the
    wrapper hyperparameters (hidden_units, dropout, epochs, patience) into
    a single .pkl manifest alongside the .keras file.
    load() restores both correctly.
    StackingEnsemble.load() detects the neural variant via the manifest key.

    Parameters
    ----------
    hidden_units : Tuple of ints — hidden layer sizes.
    dropout      : Dropout rate applied between hidden layers.
    epochs       : Max training epochs.
    patience     : Early stopping patience.
    """

    def __init__(
        self,
        hidden_units: tuple = (64, 32),
        dropout: float = 0.2,
        epochs: int = 100,
        patience: int = 10,
    ):
        self.hidden_units = hidden_units
        self.dropout      = dropout
        self.epochs       = epochs
        self.patience     = patience
        self._model       = None

    def _build(self, n_inputs: int):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout   # BatchNorm removed (unused)
        from tensorflow.keras.optimizers import AdamW
        m = Sequential(name="NeuralMeta")
        m.add(Dense(self.hidden_units[0], activation="relu", input_shape=(n_inputs,)))
        m.add(Dropout(self.dropout))
        for u in self.hidden_units[1:]:
            m.add(Dense(u, activation="relu"))
            m.add(Dropout(self.dropout))
        m.add(Dense(1))
        m.compile(
            optimizer=AdamW(1e-3, weight_decay=1e-4),
            loss="huber",
            metrics=["mae"],
        )
        return m

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> "NeuralMetaLearner":
        # No bare `import tensorflow as tf` needed here; _build handles all TF imports.
        from tensorflow.keras.callbacks import EarlyStopping
        self._model = self._build(X.shape[1])
        self._model.fit(
            X, y,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=[
                EarlyStopping("val_loss", patience=self.patience,
                              restore_best_weights=True)
            ],
            verbose=0,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X, verbose=0).flatten()

    def save(self, path: str) -> None:
        """
        FIX (Bug 2): persist wrapper metadata AND Keras weights.
        Saves a .pkl manifest at `path` and a .keras file alongside it.
        """
        if self._model is None:
            raise RuntimeError("Model not trained; nothing to save.")
        keras_path = path.replace(".pkl", "_neural_meta.keras")
        self._model.save(keras_path)
        manifest = {
            "__type__": "NeuralMetaLearner",
            "hidden_units": self.hidden_units,
            "dropout":      self.dropout,
            "epochs":       self.epochs,
            "patience":     self.patience,
            "keras_path":   keras_path,
        }
        joblib.dump(manifest, path)
        log.info(f"NeuralMetaLearner saved: manifest={path}  keras={keras_path}")

    @staticmethod
    def load(path: str) -> "NeuralMetaLearner":
        """
        FIX (Bug 2): restore wrapper hyperparameters AND Keras model.
        """
        import tensorflow as tf
        manifest = joblib.load(path)
        if not isinstance(manifest, dict) or manifest.get("__type__") != "NeuralMetaLearner":
            raise ValueError(f"File at {path} is not a NeuralMetaLearner manifest.")
        obj = NeuralMetaLearner(
            hidden_units=manifest["hidden_units"],
            dropout=manifest["dropout"],
            epochs=manifest["epochs"],
            patience=manifest["patience"],
        )
        obj._model = tf.keras.models.load_model(manifest["keras_path"])
        log.info(f"NeuralMetaLearner loaded from {path}")
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

    def __init__(self, meta_learner: str = "ridge", ridge_alpha: float = 1.0):
        self.meta_learner_type = meta_learner
        self.ridge_alpha       = ridge_alpha
        self._meta             = None
        self.val_maes_: Dict[str, float] = {}

    def fit(
        self,
        meta_X: np.ndarray,
        meta_y: np.ndarray,
        val_maes: Optional[Dict[str, float]] = None,
    ) -> "StackingEnsemble":
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
        """
        FIX (Bug 2): NeuralMetaLearner is saved via its own save() method
        which writes a .pkl manifest + a .keras file.
        All other meta-learner types are pickled directly.
        """
        if isinstance(self._meta, NeuralMetaLearner):
            # Save the NeuralMetaLearner manifest; save the ensemble shell separately
            self._meta.save(path)
            # Also persist the StackingEnsemble shell (without _meta) for reload
            shell = StackingEnsemble(
                meta_learner=self.meta_learner_type,
                ridge_alpha=self.ridge_alpha,
            )
            shell.val_maes_ = self.val_maes_
            joblib.dump(shell, path.replace(".pkl", "_shell.pkl"))
        else:
            joblib.dump(self, path)
        log.info(f"StackingEnsemble saved to {path}")

    @staticmethod
    def load(path: str) -> "StackingEnsemble":
        """
        FIX (Bug 2): detect neural variant via manifest __type__ key and
        restore correctly; fall back to direct joblib load for ridge/weighted.
        """
        data = joblib.load(path)

        # Neural variant: manifest dict was saved by NeuralMetaLearner.save()
        if isinstance(data, dict) and data.get("__type__") == "NeuralMetaLearner":
            shell_path = path.replace(".pkl", "_shell.pkl")
            obj = joblib.load(shell_path) if joblib.os.path.exists(shell_path) \
                  else StackingEnsemble(meta_learner="neural")
            obj._meta = NeuralMetaLearner.load(path)
            return obj

        # Ridge / weighted: direct joblib pickle
        return data
