"""
src/models/deep_learning.py
============================
Keras deep learning model builders: GRU, LSTM, BiLSTM+Attention,
Transformer (with residuals), Gated TFT, and TCN (Temporal Convolutional Network).

Improvements over v3.1:
  - TCN model added (dilated causal convolutions, proven strong on time-series)
  - Positional encoding added to Transformer (fixes token-order blindness)
  - MC-Dropout support for uncertainty estimation at inference time
  - AdamW optimizer replaces Adam for better weight regularisation
  - Cosine-annealing LR schedule replaces fixed ReduceLROnPlateau
  - Shared get_callbacks factory updated accordingly

All models use Huber loss.
"""

from __future__ import annotations
import os
import math
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, GRU, LSTM, Bidirectional,
    MultiHeadAttention, LayerNormalization, Dropout,
    GlobalAveragePooling1D, Conv1D, Add,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import AdamW

log = logging.getLogger(__name__)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(name: str, output_dir: str, patience: int = 7,
                  epochs: int = 40) -> list:
    """
    Return training callbacks:
      - EarlyStopping (restores best weights)
      - ModelCheckpoint
      - CosineDecayRestarts LR schedule (via LearningRateScheduler)

    Parameters
    ----------
    name       : Model name used for checkpoint filename.
    output_dir : Directory to save the best checkpoint.
    patience   : Early stopping patience (epochs).
    epochs     : Total epochs (used to set cosine decay period).
    """
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
        self.query = Dense(units, activation="tanh")
        self.weight = Dense(1)

    def call(self, x):
        scores = self.weight(self.query(x))              # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)          # (B, T, 1)
        return tf.reduce_sum(x * weights, axis=1)        # (B, units)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).
    Adds temporal order signal to the Transformer input — without this
    the self-attention is permutation-invariant and ignores time order.
    """

    def __init__(self, max_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model  = tf.shape(x)[2]
        # Build PE on-the-fly in float32
        positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)   # (T,1)
        dims      = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)   # (1,D)
        angle_rates = 1.0 / tf.pow(
            10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32)
        )
        angles = positions * angle_rates                                      # (T,D)
        # Apply sin to even indices, cos to odd
        sin_part = tf.math.sin(angles[:, 0::2])
        cos_part = tf.math.cos(angles[:, 1::2])
        # Interleave
        pe = tf.reshape(
            tf.stack([sin_part, cos_part], axis=2),
            [seq_len, -1]
        )[:, :d_model]                                                        # (T,D)
        return x + tf.cast(pe[tf.newaxis, :, :], x.dtype)                    # (B,T,D)


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
        # Pre-LN: normalise BEFORE sublayer (more stable than post-LN)
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
    Uses dilated causal convolutions + weight normalisation (via kernel
    constraint) + residual connection. Proven competitive with LSTMs
    on sequence modelling benchmarks while being faster to train.

    Reference: "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling" (Bai et al., 2018).
    """

    def __init__(self, filters: int, kernel_size: int, dilation: int,
                 dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv1D(
            filters, kernel_size,
            dilation_rate=dilation, padding="causal", activation="relu",
        )
        self.conv2 = Conv1D(
            filters, kernel_size,
            dilation_rate=dilation, padding="causal", activation="relu",
        )
        self.drop1  = Dropout(dropout)
        self.drop2  = Dropout(dropout)
        self.ln1    = LayerNormalization(epsilon=1e-6)
        self.ln2    = LayerNormalization(epsilon=1e-6)
        # 1×1 projection when channel dims differ
        self.proj   = None
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
    """Two-layer GRU with AdamW."""
    inp = Input(shape=(timesteps, n_features))
    x = GRU(128, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = GRU(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="GRU")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_lstm(timesteps: int, n_features: int) -> Model:
    """Stacked LSTM with AdamW."""
    inp = Input(shape=(timesteps, n_features))
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="LSTM")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_bilstm_attention(timesteps: int, n_features: int) -> Model:
    """Bidirectional LSTM with soft attention pooling."""
    inp = Input(shape=(timesteps, n_features))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    ctx = AttentionPool(64)(x)
    x = Dense(32, activation="relu")(ctx)
    out = Dense(1)(x)
    m = Model(inp, out, name="BiLSTM_Attn")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_transformer(timesteps: int, n_features: int) -> Model:
    """
    Transformer encoder with:
      - Positional encoding (fixes permutation-invariance)
      - Pre-LN residual blocks (more stable than post-LN)
      - AdamW optimiser
    """
    d_model = min(n_features, 64)
    n_heads = max(1, d_model // 16)
    inp = Input(shape=(timesteps, n_features))
    x = Dense(d_model)(inp)
    x = PositionalEncoding()(x)                      # <-- NEW
    x = TransformerBlock(n_heads, d_model, d_model * 2)(x)
    x = TransformerBlock(n_heads, d_model, d_model * 2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="Transformer")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_tft(timesteps: int, n_features: int) -> Model:
    """Gated TFT with Pre-LN and AdamW."""
    d_model = min(n_features, 64)
    n_heads = max(1, d_model // 16)
    inp = Input(shape=(timesteps, n_features))
    x = Dense(d_model)(inp)
    x = PositionalEncoding()(x)                      # <-- NEW
    x = GatedTFTBlock(n_heads, d_model, d_model * 2)(x)
    x = GatedTFTBlock(n_heads, d_model, d_model * 2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="TFT")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


def build_tcn(timesteps: int, n_features: int,
              filters: int = 64, kernel_size: int = 3,
              num_blocks: int = 4) -> Model:
    """
    Temporal Convolutional Network (Bai et al., 2018).

    Uses exponentially increasing dilations (1, 2, 4, 8, ...) so that
    the receptive field grows as 2^num_blocks * (kernel_size - 1),
    covering the full lookback window without depth blowup.

    Parameters
    ----------
    timesteps  : Sequence length (lookback window).
    n_features : Number of input features per timestep.
    filters    : Number of convolutional filters per block.
    kernel_size: Kernel size for dilated convolutions.
    num_blocks : Number of TCN residual blocks (receptive field = 2^num_blocks * (k-1)).
    """
    inp = Input(shape=(timesteps, n_features))
    x   = inp
    for i in range(num_blocks):
        x = TCNBlock(filters, kernel_size, dilation=2 ** i)(x)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation="relu")(x)
    x   = Dropout(0.2)(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name="TCN")
    m.compile(optimizer=_OPT(), loss=Huber(), metrics=["mae"])
    return m


# ── MC-Dropout Inference Helper ───────────────────────────────────────────────

def mc_predict(model: Model, X: "np.ndarray",
               n_samples: int = 30,
               batch_size: int = 256) -> tuple:
    """
    Monte-Carlo Dropout inference: run the model n_samples times with
    dropout active to estimate predictive mean and uncertainty (std dev).

    Parameters
    ----------
    model     : Trained Keras model (must contain Dropout layers).
    X         : Input array of shape (N, timesteps, features).
    n_samples : Number of stochastic forward passes.
    batch_size: Batch size for each forward pass.

    Returns
    -------
    mean_pred : np.ndarray of shape (N,) — averaged predictions.
    std_pred  : np.ndarray of shape (N,) — std dev across samples (uncertainty).
    """
    import numpy as np
    preds = np.stack(
        [model(X, training=True).numpy().flatten() for _ in range(n_samples)],
        axis=0,
    )                   # (n_samples, N)
    return preds.mean(axis=0), preds.std(axis=0)


# ── Builder Registry ──────────────────────────────────────────────────────────

DL_BUILDERS = {
    "GRU":          build_gru,
    "LSTM":         build_lstm,
    "BiLSTM-Attn":  build_bilstm_attention,
    "Transformer":  build_transformer,
    "TFT":          build_tft,
    "TCN":          build_tcn,      # NEW
}
"""
Registry of all available deep learning model builder functions.
Keys are display names; values are callables (timesteps, n_features) -> Model.
"""
