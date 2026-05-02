"""
Advanced Multi-Factor Forex Prediction System v2
=================================================
New in v2:
  - XGBoost + LightGBM tree models alongside deep learning
  - Stacking ensemble (meta-learner on all model outputs)
  - Huber loss (robust to outliers) for deep learning
  - Per-currency IQR outlier removal (fixes global distortion)
  - Extended features: Stochastic RSI, Williams %R, OBV, lags, calendar
  - SHAP feature importance for best tree model
  - Residual connections in Transformer & TFT
  - Improved visualizations with seaborn
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "Forex_Data.csv"
OUTPUT_DIR   = "outputs"
TIMESTEPS    = 15        # longer lookback
TEST_RATIO   = 0.2
BATCH_SIZE   = 256
EPOCHS       = 40
PATIENCE     = 7
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

# ── 1. Load & Clean ───────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading and Cleaning Data")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
df = df.drop(columns=["currency"], errors="ignore").dropna()
print(f"  Raw shape: {df.shape}")

from tqdm import tqdm

# Per-currency IQR outlier removal (fixes global distortion)
cleaned = []
for code, grp in tqdm(df.groupby("currency_code"), desc="Removing Outliers"):
    q1, q3 = grp["exchange_rate"].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (grp["exchange_rate"] >= q1 - 1.5*iqr) & (grp["exchange_rate"] <= q3 + 1.5*iqr)
    cleaned.append(grp[mask])
df = pd.concat(cleaned, ignore_index=True)
df = df.sort_values(["currency_code", "date"]).reset_index(drop=True)
print(f"  After per-currency IQR: {df.shape}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Feature Engineering")
print("=" * 70)

def add_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    p = g["exchange_rate"]

    # Moving averages
    for w in [5, 10, 20, 50]:
        g[f"SMA_{w}"] = p.rolling(w, min_periods=1).mean()
    for w in [5, 12, 26]:
        g[f"EMA_{w}"] = p.ewm(span=w, adjust=False).mean()

    # MACD
    g["MACD"] = g["EMA_12"] - g["EMA_26"]
    g["MACD_signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
    g["MACD_hist"] = g["MACD"] - g["MACD_signal"]

    # RSI-14
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    g["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # Stochastic RSI
    rsi = g["RSI"]
    rsi_min = rsi.rolling(14, min_periods=1).min()
    rsi_max = rsi.rolling(14, min_periods=1).max()
    g["StochRSI"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    # Williams %R
    high14 = p.rolling(14, min_periods=1).max()
    low14  = p.rolling(14, min_periods=1).min()
    g["WilliamsR"] = -100 * (high14 - p) / (high14 - low14 + 1e-10)

    # Bollinger Bands
    sma20 = p.rolling(20, min_periods=1).mean()
    std20 = p.rolling(20, min_periods=1).std().fillna(0)
    g["BB_upper"] = sma20 + 2*std20
    g["BB_lower"] = sma20 - 2*std20
    g["BB_width"] = g["BB_upper"] - g["BB_lower"]
    g["BB_pct"]   = (p - g["BB_lower"]) / (g["BB_width"] + 1e-10)

    # ATR
    if {"high", "low", "close"}.issubset(g.columns):
        tr = pd.concat([
            g["high"] - g["low"],
            (g["high"] - g["close"].shift()).abs(),
            (g["low"]  - g["close"].shift()).abs()
        ], axis=1).max(axis=1)
        g["ATR"] = tr.rolling(14, min_periods=1).mean()
        g["ATR_pct"] = g["ATR"] / (p + 1e-10)

    # OBV (proxy using volume and direction)
    if "volume" in g.columns:
        direction = np.sign(p.diff().fillna(0))
        g["OBV"] = (direction * g["volume"]).cumsum()

    # Returns & volatility
    g["log_return"]    = np.log(p / p.shift(1)).fillna(0)
    g["pct_change"]    = p.pct_change().fillna(0)
    g["volatility_5"]  = g["log_return"].rolling(5,  min_periods=1).std().fillna(0)
    g["volatility_20"] = g["log_return"].rolling(20, min_periods=1).std().fillna(0)
    g["vol_ratio"]     = g["volatility_5"] / (g["volatility_20"] + 1e-10)

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        g[f"lag_{lag}"] = p.shift(lag).bfill()

    # Calendar features (date already parsed globally)
    g["day_of_week"]  = g["date"].dt.dayofweek
    g["month"]        = g["date"].dt.month
    g["quarter"]      = g["date"].dt.quarter
    g["is_month_end"] = g["date"].dt.is_month_end.astype(int)

    return g

dfs = []
for code, grp in tqdm(df.groupby("currency_code"), desc="Feature Engineering"):
    g = add_features(grp)
    g["currency_code"] = code
    dfs.append(g)

df = pd.concat(dfs, ignore_index=True).dropna().reset_index(drop=True)
print(f"  After feature engineering: {df.shape}")

# One-hot encode currency
df = pd.get_dummies(df, columns=["currency_code"], drop_first=True)
bool_cols = df.select_dtypes("bool").columns
if len(bool_cols):
    df[bool_cols] = df[bool_cols].astype(int)
print(f"  After encoding: {df.shape}")

# ── 3. Scale & Split ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Scaling and Splitting")
print("=" * 70)

feature_cols = [c for c in df.columns if c not in ("date", "exchange_rate")]
X_all = df[feature_cols].values.astype(np.float32)
y_all = df["exchange_rate"].values.reshape(-1, 1).astype(np.float32)

scaler_X = RobustScaler()   # RobustScaler handles outlier-heavy forex better
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_all)
y_scaled = scaler_y.fit_transform(y_all)

split = int(len(X_scaled) * (1 - TEST_RATIO))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

# Flat versions for tree models
X_train_flat, y_train_flat = X_train, y_train.ravel()
X_test_flat,  y_test_flat  = X_test,  y_test.ravel()

# Sequences for deep learning
def make_sequences(X, y, ts):
    Xs, ys = [], []
    for i in range(len(X) - ts):
        Xs.append(X[i:i+ts])
        ys.append(y[i+ts])
    return np.array(Xs), np.array(ys)

X_tr_seq, y_tr_seq = make_sequences(X_train, y_train, TIMESTEPS)
X_te_seq, y_te_seq = make_sequences(X_test,  y_test,  TIMESTEPS)
n_feat = X_tr_seq.shape[2]
print(f"  Seq train: {X_tr_seq.shape}  |  Seq test: {X_te_seq.shape}")

# ── 4. Deep Learning Models ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Building Deep Learning Models")
print("=" * 70)

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, GRU, LSTM, Bidirectional,
    MultiHeadAttention, LayerNormalization, Dropout,
    Multiply, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    [tf.config.experimental.set_memory_growth(d, True) for d in gpus]
    print(f"  GPU: {gpus}")
else:
    print("  CPU mode")

def callbacks():
    return [
        EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

# GRU with residual
def build_gru():
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = GRU(128, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = GRU(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile("adam", Huber(), metrics=["mae"])
    return m

# LSTM
def build_lstm():
    m = Sequential([
        LSTM(128, input_shape=(TIMESTEPS, n_feat), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    m.compile("adam", Huber(), metrics=["mae"])
    return m

# BiLSTM + Attention
class AttentionPool(tf.keras.layers.Layer):
    def __init__(self, u, **kw):
        super().__init__(**kw)
        self.q = Dense(u, activation="tanh")
        self.w = Dense(1)
    def call(self, x):
        sc = self.w(self.q(x))
        wt = tf.nn.softmax(sc, axis=1)
        return tf.reduce_sum(x * wt, axis=1)

def build_bilstm():
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    ctx = AttentionPool(64)(x)
    x = Dense(32, activation="relu")(ctx)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile("adam", Huber(), metrics=["mae"])
    return m

# Transformer with residual
class TBlock(tf.keras.layers.Layer):
    def __init__(self, nh, dm, ff, dr=0.1, **kw):
        super().__init__(**kw)
        self.att  = MultiHeadAttention(num_heads=nh, key_dim=dm//nh)
        self.ffn  = Sequential([Dense(ff, "relu"), Dense(dm)])
        self.ln1  = LayerNormalization(1e-6)
        self.ln2  = LayerNormalization(1e-6)
        self.d1   = Dropout(dr)
        self.d2   = Dropout(dr)
    def call(self, x, training=False):
        a = self.d1(self.att(x, x), training=training)
        x = self.ln1(x + a)
        f = self.d2(self.ffn(x),   training=training)
        return self.ln2(x + f)

def build_transformer():
    dm = min(n_feat, 64)
    nh = max(1, dm // 16)
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = Dense(dm)(inp)                    # project to d_model
    x = TBlock(nh, dm, dm*2)(x)
    x = TBlock(nh, dm, dm*2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, "relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile("adam", Huber(), metrics=["mae"])
    return m

# Gated TFT
class TFTBlock(tf.keras.layers.Layer):
    def __init__(self, nh, dm, ff, dr=0.1, **kw):
        super().__init__(**kw)
        self.att  = MultiHeadAttention(num_heads=nh, key_dim=dm//nh)
        self.gate = Dense(dm, "sigmoid")
        self.ffn  = Sequential([Dense(ff, "relu"), Dense(dm)])
        self.ln1  = LayerNormalization(1e-6)
        self.ln2  = LayerNormalization(1e-6)
        self.d1   = Dropout(dr)
        self.d2   = Dropout(dr)
    def call(self, x, training=False):
        a = self.d1(self.att(x, x), training=training)
        g = self.gate(x)
        x = self.ln1(x + a * g)
        f = self.d2(self.ffn(x),   training=training)
        return self.ln2(x + f)

def build_tft():
    dm = min(n_feat, 64)
    nh = max(1, dm // 16)
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = Dense(dm)(inp)
    x = TFTBlock(nh, dm, dm*2)(x)
    x = TFTBlock(nh, dm, dm*2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, "relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile("adam", Huber(), metrics=["mae"])
    return m

# ── 5. Tree Models ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Tree Models (XGBoost + LightGBM)")
print("=" * 70)

xgb_model = xgb.XGBRegressor(
    n_estimators=800, learning_rate=0.05, max_depth=7,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
    tree_method="hist", n_jobs=-1, verbosity=0
)
xgb_model.fit(
    X_train_flat, y_train_flat,
    eval_set=[(X_test_flat, y_test_flat)],
    verbose=False
)
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
print("  XGBoost trained & saved.")

lgb_model = lgb.LGBMRegressor(
    n_estimators=800, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
    n_jobs=-1, verbosity=-1
)
lgb_model.fit(
    X_train_flat, y_train_flat,
    eval_set=[(X_test_flat, y_test_flat)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
)
joblib.dump(lgb_model, os.path.join(OUTPUT_DIR, "lgb_model.pkl"))
print("  LightGBM trained & saved.")

# ── 6. Metrics Helper ─────────────────────────────────────────────────────────
def metrics(y_true, y_pred, name):
    mse  = float(np.mean((y_true - y_pred)**2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
    ss_r = np.sum((y_true - y_pred)**2)
    ss_t = np.sum((y_true - np.mean(y_true))**2)
    r2   = float(1 - ss_r/(ss_t+1e-10))
    da   = float(np.mean(np.sign(np.diff(y_true.flatten())) == np.sign(np.diff(y_pred.flatten())))*100) if len(y_true)>1 else 0.0
    return {"Model":name,"MSE":mse,"MAE":mae,"RMSE":rmse,"MAPE":mape,"R2":r2,"DA":da}

def inv(arr):
    return scaler_y.inverse_transform(arr.reshape(-1,1)).flatten()

# ── 7. Train & Evaluate Deep Learning ────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Training Deep Learning Models")
print("=" * 70)

dl_builders = {
    "GRU": build_gru, "LSTM": build_lstm,
    "BiLSTM-Attn": build_bilstm,
    "Transformer": build_transformer, "TFT": build_tft,
}

results, histories, preds = [], {}, {}
dl_test_preds = {}   # scaled preds for stacking

for name, fn in dl_builders.items():
    print(f"\n  Training {name}...")
    m = fn()
    h = m.fit(
        X_tr_seq, y_tr_seq,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2, callbacks=callbacks(), verbose=0
    )
    p_scaled = m.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0).flatten()
    dl_test_preds[name] = p_scaled
    y_pr = inv(p_scaled)
    y_tr = inv(y_te_seq.flatten())
    m_dict = metrics(y_tr, y_pr, name)
    results.append(m_dict)
    histories[name] = h
    preds[name] = (y_pr, y_tr)
    m.save(os.path.join(OUTPUT_DIR, f"{name.lower().replace('-','_').replace(' ','_')}.keras"))
    print(f"    MAE={m_dict['MAE']:.4f}  R²={m_dict['R2']:.4f}  DA={m_dict['DA']:.1f}%")

# Tree model evaluation (aligned to sequence test set length)
n_seq = len(y_te_seq)
X_te_aligned = X_test_flat[-n_seq:]
y_te_aligned  = y_test_flat[-n_seq:]

for tname, tmodel in [("XGBoost", xgb_model), ("LightGBM", lgb_model)]:
    p_sc = tmodel.predict(X_te_aligned).astype(np.float32)
    y_pr = inv(p_sc)
    y_tr = inv(y_te_aligned)
    m_dict = metrics(y_tr, y_pr, tname)
    results.append(m_dict)
    preds[tname] = (y_pr, y_tr)
    print(f"  {tname}: MAE={m_dict['MAE']:.4f}  R²={m_dict['R2']:.4f}  DA={m_dict['DA']:.1f}%")

# ── 8. Stacking Ensemble ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Stacking Ensemble")
print("=" * 70)

# Build meta-features from all model predictions on test set
meta_X = np.column_stack([
    *[dl_test_preds[n] for n in dl_builders],
    xgb_model.predict(X_te_aligned),
    lgb_model.predict(X_te_aligned),
])
meta_y = y_te_seq.flatten()

# Simple hold-out split inside test for meta-learner
meta_split = int(len(meta_X) * 0.5)
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_X[:meta_split], meta_y[:meta_split])
joblib.dump(meta_learner, os.path.join(OUTPUT_DIR, "stacking_meta.pkl"))

stack_pred_sc = meta_learner.predict(meta_X[meta_split:])
y_pr_stack = inv(stack_pred_sc)
y_tr_stack = inv(meta_y[meta_split:])
stack_metrics = metrics(y_tr_stack, y_pr_stack, "Stacking Ensemble")
results.append(stack_metrics)
preds["Stacking Ensemble"] = (y_pr_stack, y_tr_stack)
print(f"  Stacking: MAE={stack_metrics['MAE']:.4f}  R²={stack_metrics['R2']:.4f}  DA={stack_metrics['DA']:.1f}%")

# ── 9. SHAP Feature Importance ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: SHAP Feature Importance (LightGBM)")
print("=" * 70)

try:
    import shap
    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer.shap_values(X_te_aligned[:500])
    shap_mean = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(shap_mean)[-20:][::-1]
    top_feats = [feature_cols[i] for i in top_idx]
    top_vals  = shap_mean[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_vals, y=top_feats, ax=ax, palette="viridis")
    ax.set_title("Top 20 Features by SHAP Importance (LightGBM)", fontsize=13)
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance.png"), dpi=150)
    plt.close()
    print("  Saved shap_importance.png")
except Exception as e:
    print(f"  SHAP skipped: {e}")

# ── 10. Results Table ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

# ── 11. Visualizations ────────────────────────────────────────────────────────
print("\nGenerating Visualizations...")

# Loss curves (DL only)
fig, axes = plt.subplots(1, len(dl_builders), figsize=(5*len(dl_builders), 4), sharey=True)
for i, (nm, hist) in enumerate(histories.items()):
    ax = axes[i] if len(dl_builders) > 1 else axes
    ax.plot(hist.history["loss"],     label="Train")
    ax.plot(hist.history["val_loss"], label="Val",  linestyle="--")
    ax.set_title(nm, fontsize=10)
    ax.set_xlabel("Epoch")
    if i == 0: ax.set_ylabel("Huber Loss")
    ax.legend(fontsize=7)
plt.suptitle("Training & Validation Loss", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# Metrics heatmap
metrics_pivot = results_df.set_index("Model")[["MAE","RMSE","MAPE","R2","DA"]].astype(float)
# Normalise for heatmap readability
norm_pivot = (metrics_pivot - metrics_pivot.min()) / (metrics_pivot.max() - metrics_pivot.min() + 1e-10)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(norm_pivot, annot=metrics_pivot.round(4), fmt="", cmap="RdYlGn", ax=ax,
            linewidths=0.5, cbar_kws={"label": "Normalized Score"})
ax.set_title("Model Metrics Heatmap (values annotated, colors normalized)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_heatmap.png"), dpi=150)
plt.close()

# Actual vs Predicted
for nm, (yp, yt) in preds.items():
    n = min(300, len(yt))
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(yt[:n], label="Actual",    lw=1.5)
    ax.plot(yp[:n], label="Predicted", lw=1.0, alpha=0.85, linestyle="--")
    ax.set_title(f"{nm} – Actual vs Predicted", fontsize=11)
    ax.set_xlabel("Sample"); ax.set_ylabel("Exchange Rate")
    ax.legend()
    plt.tight_layout()
    fname = f"pred_{nm.lower().replace(' ','_').replace('-','_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()

print("  All plots saved.")

# ── 12. Summary ───────────────────────────────────────────────────────────────
best = min(results, key=lambda r: r["MAE"])
print("\n" + "=" * 70)
print(f"BEST MODEL: {best['Model']}")
for k in ["MSE","MAE","RMSE","MAPE","R2","DA"]:
    print(f"  {k:5s}: {best[k]:.6f}")
print("=" * 70)
print(f"\nOutputs → {os.path.abspath(OUTPUT_DIR)}")
print("Forex Prediction v2 complete ✓")
