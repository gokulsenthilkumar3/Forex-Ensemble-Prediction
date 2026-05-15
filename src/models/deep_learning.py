"""
src/models/deep_learning.py
============================
Keras deep learning model builders: GRU, LSTM, BiLSTM+Attention,
Transformer (with residuals), and Gated TFT.

All models use Huber loss and share a common callback factory.
"""

from __future__ import annotations
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, GRU, LSTM, Bidirectional,
    MultiHeadAttention, LayerNormalization, Dropout,
    GlobalAveragePooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import Huber

log = logging.getLogger(__name__)


def get_callbacks(name: str, output_dir: str, patience: int = 7) -> list:
    """
    Return standard training callbacks: EarlyStopping, ReduceLROnPlateau,
    and ModelCheckpoint.

    Parameters
    ----------
    name       : Model name used for checkpoint filename.
    output_dir : Directory to save the best checkpoint.
    patience   : Early stopping patience (epochs).
    """
    ckpt = os.path.join(output_dir, f"{name}_best.keras")
    return [
        EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(ckpt, save_best_only=True, monitor="val_loss", verbose=0),
    ]


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


class TransformerBlock(tf.keras.layers.Layer):
    """Multi-head self-attention block with residual connection and layer norm."""

    def __init__(self, num_heads: int, d_model: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.att  = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn  = Sequential([Dense(ff_dim, "relu"), Dense(d_model)])
        self.ln1  = LayerNormalization(epsilon=1e-6)
        self.ln2  = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)

    def call(self, x, training=False):
        x = self.ln1(x + self.drop1(self.att(x, x), training=training))
        return self.ln2(x + self.drop2(self.ffn(x), training=training))


class GatedTFTBlock(tf.keras.layers.Layer):
    """Gated Temporal Fusion Transformer block with per-step gating."""

    def __init__(self, num_heads: int, d_model: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.att   = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.gate  = Dense(d_model, activation="sigmoid")
        self.ffn   = Sequential([Dense(ff_dim, "relu"), Dense(d_model)])
        self.ln1   = LayerNormalization(epsilon=1e-6)
        self.ln2   = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)

    def call(self, x, training=False):
        attn = self.drop1(self.att(x, x), training=training)
        x = self.ln1(x + attn * self.gate(x))
        return self.ln2(x + self.drop2(self.ffn(x), training=training))


def build_gru(timesteps: int, n_features: int) -> Model:
    """
    Build a two-layer GRU regression model.

    Parameters
    ----------
    timesteps  : Sequence length (lookback window).
    n_features : Number of input features per timestep.
    """
    inp = Input(shape=(timesteps, n_features))
    x = GRU(128, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = GRU(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="GRU")
    m.compile(optimizer="adam", loss=Huber(), metrics=["mae"])
    return m


def build_lstm(timesteps: int, n_features: int) -> Model:
    """
    Build a stacked LSTM regression model.

    Parameters
    ----------
    timesteps  : Sequence length.
    n_features : Number of input features.
    """
    m = Sequential([
        LSTM(128, input_shape=(timesteps, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ], name="LSTM")
    m.compile(optimizer="adam", loss=Huber(), metrics=["mae"])
    return m


def build_bilstm_attention(timesteps: int, n_features: int) -> Model:
    """
    Build a Bidirectional LSTM model with soft attention pooling.

    Parameters
    ----------
    timesteps  : Sequence length.
    n_features : Number of input features.
    """
    inp = Input(shape=(timesteps, n_features))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    ctx = AttentionPool(64)(x)
    x = Dense(32, activation="relu")(ctx)
    out = Dense(1)(x)
    m = Model(inp, out, name="BiLSTM_Attn")
    m.compile(optimizer="adam", loss=Huber(), metrics=["mae"])
    return m


def build_transformer(timesteps: int, n_features: int) -> Model:
    """
    Build a Transformer encoder model with two residual blocks.

    Parameters
    ----------
    timesteps  : Sequence length.
    n_features : Number of input features.
    """
    d_model = min(n_features, 64)
    n_heads = max(1, d_model // 16)
    inp = Input(shape=(timesteps, n_features))
    x = Dense(d_model)(inp)
    x = TransformerBlock(n_heads, d_model, d_model * 2)(x)
    x = TransformerBlock(n_heads, d_model, d_model * 2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="Transformer")
    m.compile(optimizer="adam", loss=Huber(), metrics=["mae"])
    return m


def build_tft(timesteps: int, n_features: int) -> Model:
    """
    Build a Gated Temporal Fusion Transformer model.

    Parameters
    ----------
    timesteps  : Sequence length.
    n_features : Number of input features.
    """
    d_model = min(n_features, 64)
    n_heads = max(1, d_model // 16)
    inp = Input(shape=(timesteps, n_features))
    x = Dense(d_model)(inp)
    x = GatedTFTBlock(n_heads, d_model, d_model * 2)(x)
    x = GatedTFTBlock(n_heads, d_model, d_model * 2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="TFT")
    m.compile(optimizer="adam", loss=Huber(), metrics=["mae"])
    return m


DL_BUILDERS = {
    "GRU":          build_gru,
    "LSTM":         build_lstm,
    "BiLSTM-Attn":  build_bilstm_attention,
    "Transformer":  build_transformer,
    "TFT":          build_tft,
}
"""
Registry of all available deep learning model builder functions.
Keys are display names; values are callables (timesteps, n_features) -> Model.
"""
