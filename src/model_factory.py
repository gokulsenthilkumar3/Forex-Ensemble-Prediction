import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, GRU, LSTM, Bidirectional,
    MultiHeadAttention, LayerNormalization, Dropout,
    GlobalAveragePooling1D
)
from tensorflow.keras.losses import Huber

class ModelFactory:
    @staticmethod
    def build_transformer(timesteps, n_feat):
        dm = min(n_feat, 64)
        nh = max(1, dm // 16)
        inp = Input(shape=(timesteps, n_feat))
        x = Dense(dm)(inp)
        
        # Transformer Block
        attn = MultiHeadAttention(num_heads=nh, key_dim=dm//nh)(x, x)
        x = LayerNormalization(1e-6)(x + Dropout(0.1)(attn))
        
        ffn = Sequential([Dense(dm*2, "relu"), Dense(dm)])(x)
        x = LayerNormalization(1e-6)(x + Dropout(0.1)(ffn))
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, "relu")(x)
        out = Dense(1)(x)
        
        m = Model(inp, out)
        m.compile("adam", Huber(), metrics=["mae"])
        return m

    @staticmethod
    def build_bilstm_attn(timesteps, n_feat):
        inp = Input(shape=(timesteps, n_feat))
        x = Bidirectional(LSTM(64, return_sequences=True))(inp)
        x = Dropout(0.2)(x)
        
        # Simple Attention Pooling
        q = Dense(64, activation="tanh")(x)
        wt = tf.nn.softmax(Dense(1)(q), axis=1)
        ctx = tf.reduce_sum(x * wt, axis=1)
        
        x = Dense(32, activation="relu")(ctx)
        out = Dense(1)(x)
        
        m = Model(inp, out)
        m.compile("adam", Huber(), metrics=["mae"])
        return m
