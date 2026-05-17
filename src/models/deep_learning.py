"""
src/models/deep_learning.py
============================
Keras deep learning model builders: GRU, LSTM, BiLSTM+Attention,
Transformer (with residuals), Gated TFT, and TCN.

Fixed bugs (see review comment on PR #6):
  - Bug 4: PositionalEncoding rewritten using even/odd mask approach —
            avoids tf.stack shape mismatch when d_model is odd.
  - Bug 5: build_tcn now projects input to `filters` dims via Dense(filters)
            before the TCN stack, preventing aggressive one-shot downsampling.
  - Minor: unused `math` import removed.
"""

from __future__ import annotations
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, GRU, LSTM, Bidirectional,
    MultiHeadAttention, LayerNormalization, Dropout,
    GlobalAveragePooling1D, Conv1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import AdamW

log = logging.getLogger(__name__)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(
    name: str,
    output_dir: str,
    patience: int = 7,
    epochs: int = 40,
) -> list:
    ckpt = os.path.join(output_dir, f"{name}_best.keras")
    cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=max(1, epochs // 2),
        t_mul=1.0,
        m_mul=0.9,
        alpha=1e-6,
    )
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: float(cosine_decay(epoch)), verbose=0
    )
    return [
        EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(ckpt, save_best_only=True, monitor="val_loss", verbose=0),
        lr_scheduler,
    ]


# ── Custom Layers ─────────────────────────────────────────────────────────────

class AttentionPool(tf.keras.layers.Layer):
    """Soft attention pooling over the sequence (time) dimension."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.query  = Dense(units, activation="tanh")
        self.weight = Dense(1)

    def call(self, x):
        scores  = self.weight(self.query(x))         # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)      # (B, T, 1)
        return tf.reduce_sum(x * weights, axis=1)    # (B, units)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    FIX (Bug 4): Previous implementation split angles into sin_part /
    cos_part and used tf.stack, which requires equal last-dimension sizes.
    When d_model is odd, sin_part has ceil(d_model/2) columns and cos_part
    has floor(d_model/2) columns — tf.stack raises a shape error.

    New approach: build the full angle matrix, create even/odd boolean masks,
    and apply sin/cos in-place via masking. Works for any d_model.
    """

    def __init__(self, max_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model  = tf.shape(x)[2]

        positions   = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)  # (T,1)
        dims        = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)  # (1,D)
        angle_rates = 1.0 / tf.pow(
            10000.0,
            (2.0 * tf.math.floor(dims / 2.0)) / tf.cast(d_model, tf.float32),
        )
        angles = positions * angle_rates                                       # (T,D)

        # FIX: use even/odd masks instead of slicing + tf.stack.
        # mask_even[d] == 1 when d is even (apply sin), 0 when odd (apply cos).
        mask_even = tf.cast(tf.range(d_model) % 2 == 0, tf.float32)          # (D,)
        mask_odd  = 1.0 - mask_even

        pe = tf.math.sin(angles) * mask_even + tf.math.cos(angles) * mask_odd  # (T,D)
        return x + tf.cast(pe[tf.newaxis, :, :], x.dtype)                      # (B,T,D)


class TransformerBlock(tf.keras.layers.Layer):
    """Multi-head self-attention block with Pre-LN (more stable training)."""

    def __init__(self, num_heads: int, d_model: int, ff_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.att   = MultiHeadAttention(num_heads=num_heads,
                                        key_dim=d_model // num_heads)
        self.ffn   = Sequential([Dense(ff_dim, "relu"), Dense(d_model)])
        self.ln1   = LayerNormalization(epsilon=1e-6)
        self.ln2   = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)

    def call(self, x, training=False):
        x2 = self.ln1(x)
        x  = x + self.drop1(self.att(x2, x2), training=training)
        x2 = self.ln2(x)
        return x + self.drop2(self.ffn(x2), training=training)


class GatedTFTBlock(tf.keras.layers.Layer):
    """Gated Temporal Fusion Transformer block with per-step gating."""

    def __init__(self, num_heads: int, d_model: int, ff_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.att   = MultiHeadAttention(num_heads=num_heads,
                                        key_dim=d_model // num_heads)
        self.gate  = Dense(d_model, activation="sigmoid")
        self.ffn   = Sequential([Dense(ff_dim, "relu"), Dense(d_model)])
        self.ln1   = LayerNormalization(epsilon=1e-6)
        self.ln2   = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)

    def call(self, x, training=False):
        x2   = self.ln1(x)
        attn = self.drop1(self.att(x2, x2), training=training)
        x    = x + attn * self.gate(x)
        x2   = self.ln2(x)
        return x + self.drop2(self.ffn(x2), training=training)


class TCNBlock(tf.keras.layers.Layer):
    """
    Temporal Convolutional Network residual block (Bai et al. 2018).
    Uses dilated causal convolutions + residual connection.
    The 1x1 projection is only built when the residual channel count differs.
    """

    def __init__(self, filters: int, kernel_size: int, dilation: int,
                 dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv1    = Conv1D(filters, kernel_size, dilation_rate=dilation,
                               padding="causal", activation="relu")
        self.conv2    = Conv1D(filters, kernel_size, dilation_rate=dilation,
                               padding="causal", activation="relu")
        self.drop1    = Dropout(dropout)
        self.drop2    = Dropout(dropout)
        self.ln1      = LayerNormalization(epsilon=1e-6)
        self.ln2      = LayerNormalization(epsilon=1e-6)
        self.proj     = None
        self._filters = filters

    def build(self, input_shape):
        if input_shape[-1] != self._filters:
            self.proj = Conv1D(self._filters, 1, padding="same")
        super().build(input_shape)

    def call(self, x, training=False):
        residual = self.proj(x) if self.proj else x
        out = self.drop1(self.ln1(self.conv1(x)), training=training)
        out = self.drop2(self.ln2(self.conv2(out)), training=training)
        return tf.keras.activations.relu(out + residual)


# ── Model Builders ────────────────────────────────────────────────────────────

_OPT = lambda: AdamW(learning_rate=1e-3, weight_decay=1e-4)


def build_gru(timesteps: int, n_features: int) -> Model:
    inp = Input(shape=(timesteps, n_features))
    x   = GRU(128, return_sequences=True)(inp)
    x   = Dropout(0.2)(x)
    x   = GRU(64)(x)
    x   = Dropout(0.2)(x)
    x   = Dense(32, activation="relu")(x)
    m   = Model(inp, Dense(1)(x), name="GRU")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_lstm(timesteps: int, n_features: int) -> Model:
    inp = Input(shape=(timesteps, n_features))
    x   = LSTM(128, return_sequences=True)(inp)
    x   = Dropout(0.2)(x)
    x   = LSTM(64)(x)
    x   = Dropout(0.2)(x)
    x   = Dense(32, activation="relu")(x)
    m   = Model(inp, Dense(1)(x), name="LSTM")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_bilstm_attention(timesteps: int, n_features: int) -> Model:
    inp = Input(shape=(timesteps, n_features))
    x   = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x   = Dropout(0.2)(x)
    x   = Bidirectional(LSTM(32, return_sequences=True))(x)
    x   = Dropout(0.2)(x)
    ctx = AttentionPool(64)(x)
    x   = Dense(32, activation="relu")(ctx)
    m   = Model(inp, Dense(1)(x), name="BiLSTM_Attn")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_transformer(timesteps: int, n_features: int) -> Model:
    d_model = min(n_features, 64)
    n_heads = max(1, d_model // 16)
    inp = Input(shape=(timesteps, n_features))
    x   = Dense(d_model)(inp)
    x   = PositionalEncoding()(x)
    x   = TransformerBlock(n_heads, d_model, d_model * 2)(x)
    x   = TransformerBlock(n_heads, d_model, d_model * 2)(x)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation="relu")(x)
    x   = Dropout(0.2)(x)
    m   = Model(inp, Dense(1)(x), name="Transformer")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_tft(timesteps: int, n_features: int) -> Model:
    d_model = min(n_features, 64)
    n_heads = max(1, d_model // 16)
    inp = Input(shape=(timesteps, n_features))
    x   = Dense(d_model)(inp)
    x   = PositionalEncoding()(x)
    x   = GatedTFTBlock(n_heads, d_model, d_model * 2)(x)
    x   = GatedTFTBlock(n_heads, d_model, d_model * 2)(x)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation="relu")(x)
    x   = Dropout(0.2)(x)
    m   = Model(inp, Dense(1)(x), name="TFT")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_tcn(
    timesteps: int,
    n_features: int,
    filters: int = 64,
    kernel_size: int = 3,
    num_blocks: int = 4,
) -> Model:
    """
    Temporal Convolutional Network (Bai et al., 2018).

    FIX (Bug 5): Added Dense(filters) projection BEFORE the TCN stack.
    Without it, block 0 receives n_features channels (e.g. 150) and its
    1x1 residual projection compresses them to `filters=64` in one hard step.
    The Dense projection provides a smoother, learnable dimensionality
    reduction before the dilated convolutions begin.
    """
    inp = Input(shape=(timesteps, n_features))
    x   = Dense(filters)(inp)              # smooth projection to filter space
    for i in range(num_blocks):
        x = TCNBlock(filters, kernel_size, dilation=2 ** i)(x)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation="relu")(x)
    x   = Dropout(0.2)(x)
    m   = Model(inp, Dense(1)(x), name="TCN")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


# ── MC-Dropout Inference Helper ───────────────────────────────────────────────

def mc_predict(
    model: Model,
    X: "np.ndarray",
    n_samples: int = 30,
    batch_size: int = 256,
) -> tuple:
    """
    Monte-Carlo Dropout inference.

    FIX (Minor): batch_size is now actually used. Each forward pass processes
    X in chunks of batch_size to avoid OOM on large inputs.
    Uses model(batch, training=True) (not model.predict) to keep Dropout active.

    Returns
    -------
    mean_pred : np.ndarray (N,)
    std_pred  : np.ndarray (N,) — uncertainty estimate
    """
    import numpy as np

    def _single_pass(m, X_all, bs):
        """Run one stochastic forward pass over X in batches."""
        parts = []
        for start in range(0, len(X_all), bs):
            batch = X_all[start: start + bs]
            parts.append(m(batch, training=True).numpy().flatten())
        return np.concatenate(parts)

    preds = np.stack(
        [_single_pass(model, X, batch_size) for _ in range(n_samples)],
        axis=0,
    )  # (n_samples, N)
    return preds.mean(axis=0), preds.std(axis=0)


# ── Builder Registry ──────────────────────────────────────────────────────────

DL_BUILDERS = {
    "GRU":         build_gru,
    "LSTM":        build_lstm,
    "BiLSTM-Attn": build_bilstm_attention,
    "Transformer": build_transformer,
    "TFT":         build_tft,
    "TCN":         build_tcn,
}
