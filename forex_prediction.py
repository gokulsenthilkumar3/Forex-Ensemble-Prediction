"""
Advanced Multi-Factor Forex Prediction System v3.1
===================================================
Data & Feature improvements (v3.1 branch: improve/data-and-features):
  - Per-currency RobustScaler (prevents JPY/high-rate currency scale domination)
  - Config-driven feature engineering via config/features.yaml
  - Cross-currency correlation features + synthetic FX index
  - SHAP-based feature ranking exported to features_ranked.csv
  - ATR/OBV disabled by default (no OHLCV in base dataset)
  - ffill on lag features (no future data leakage)
  - Flexible target mode: raw_rate | log_return | pct_change | n_step_ahead
"""

import os, warnings, logging
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb
import lightgbm as lgb

# ── Project imports ────────────────────────────────────────────────────────────
from src.features.engineer import add_features, load_config
from src.features.cross_currency import add_cross_currency_features
from src.features.shap_ranking import compute_shap_ranking

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH    = os.environ.get("DATA_PATH",  "Forex_Data.csv")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "outputs")
TIMESTEPS    = 15
TEST_RATIO   = 0.2
BATCH_SIZE   = 256
EPOCHS       = 40
PATIENCE     = 7
RANDOM_STATE = 42
N_SPLITS     = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

# Load feature config
FEAT_CFG = load_config()
TARGET_MODE = FEAT_CFG["target"]["mode"]      # raw_rate | log_return | pct_change | n_step_ahead
N_AHEAD     = FEAT_CFG["target"]["n_steps_ahead"]
log.info(f"Feature config loaded. Target mode: {TARGET_MODE}")

# ── 1. Load & Clean ────────────────────────────────────────────────────────────
log.info("STEP 1: Loading and Cleaning Data")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
df = df.drop(columns=["currency"], errors="ignore").dropna()
log.info(f"Raw shape: {df.shape}")

from tqdm import tqdm

cleaned = []
for code, grp in tqdm(df.groupby("currency_code"), desc="Removing Outliers"):
    q1, q3 = grp["exchange_rate"].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (
        (grp["exchange_rate"] >= q1 - 1.5 * iqr) &
        (grp["exchange_rate"] <= q3 + 1.5 * iqr)
    )
    cleaned.append(grp[mask])
df = pd.concat(cleaned, ignore_index=True)
df = df.sort_values(["currency_code", "date"]).reset_index(drop=True)
log.info(f"After per-currency IQR: {df.shape}")

# ── 2. Per-Currency Feature Engineering ───────────────────────────────────────
log.info("STEP 2: Feature Engineering (config-driven, per currency)")

dfs = []
for code, grp in tqdm(df.groupby("currency_code"), desc="Feature Engineering"):
    g = add_features(grp, FEAT_CFG)
    g["currency_code"] = code
    dfs.append(g)

df = pd.concat(dfs, ignore_index=True).sort_values(["currency_code", "date"]).reset_index(drop=True)
log.info(f"After per-currency feature engineering: {df.shape}")

# ── 3. Cross-Currency Features ─────────────────────────────────────────────────
log.info("STEP 3: Cross-Currency Features")
df = add_cross_currency_features(df, FEAT_CFG["cross_currency"])
df = df.dropna().reset_index(drop=True)
log.info(f"After cross-currency features: {df.shape}")

# ── 4. Target Engineering ──────────────────────────────────────────────────────
log.info(f"STEP 4: Target Engineering (mode={TARGET_MODE})")

if TARGET_MODE == "raw_rate":
    df["target"] = df["exchange_rate"]
elif TARGET_MODE == "log_return":
    df["target"] = df.groupby("currency_code")["exchange_rate"].transform(
        lambda p: np.log(p / p.shift(1))
    )
elif TARGET_MODE == "pct_change":
    df["target"] = df.groupby("currency_code")["exchange_rate"].transform(
        lambda p: p.pct_change()
    )
elif TARGET_MODE == "n_step_ahead":
    df["target"] = df.groupby("currency_code")["exchange_rate"].transform(
        lambda p: p.shift(-N_AHEAD)
    )
else:
    raise ValueError(f"Unknown target mode: {TARGET_MODE}")

df = df.dropna(subset=["target"]).reset_index(drop=True)
log.info(f"After target engineering: {df.shape}")

# ── 5. Per-Currency Scaling ────────────────────────────────────────────────────
log.info("STEP 5: Per-Currency Scaling")

feature_cols = [
    c for c in df.columns
    if c not in ("date", "exchange_rate", "target", "currency_code")
]

# Fit one RobustScaler per currency BEFORE one-hot encoding
per_currency_scalers = {}
scaled_dfs = []
for code, grp in tqdm(df.groupby("currency_code"), desc="Fitting per-currency scalers"):
    sc = RobustScaler()
    cols_present = [c for c in feature_cols if c in grp.columns]
    grp = grp.copy()
    grp[cols_present] = sc.fit_transform(grp[cols_present].values)
    per_currency_scalers[code] = {"scaler": sc, "cols": cols_present}
    scaled_dfs.append(grp)

df = pd.concat(scaled_dfs, ignore_index=True).sort_values(["currency_code", "date"]).reset_index(drop=True)
joblib.dump(per_currency_scalers, os.path.join(OUTPUT_DIR, "per_currency_scalers.pkl"))
log.info(f"Per-currency scalers saved ({len(per_currency_scalers)} currencies).")

# One-hot encode currency_code
df = pd.get_dummies(df, columns=["currency_code"], drop_first=True)
bool_cols = df.select_dtypes("bool").columns
if len(bool_cols):
    df[bool_cols] = df[bool_cols].astype(int)
log.info(f"After one-hot encoding: {df.shape}")

# ── 6. Target Scaling ─────────────────────────────────────────────────────────
log.info("STEP 6: Target Scaling")

feature_cols_final = [c for c in df.columns if c not in ("date", "exchange_rate", "target")]
X_all = df[feature_cols_final].values.astype(np.float32)
y_all = df["target"].values.reshape(-1, 1).astype(np.float32)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_all)
joblib.dump(scaler_y, os.path.join(OUTPUT_DIR, "scaler_y.pkl"))
log.info("Target scaler saved.")

# Train/test split
split       = int(len(X_all) * (1 - TEST_RATIO))
X_train     = X_all[:split];   X_test     = X_all[split:]
y_train     = y_scaled[:split]; y_test     = y_scaled[split:]
X_train_flat, y_train_flat = X_train, y_train.ravel()
X_test_flat,  y_test_flat  = X_test,  y_test.ravel()
log.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# Sequences for deep learning
def make_sequences(X, y, ts):
    """Create sliding window sequences for time series deep learning models."""
    Xs, ys = [], []
    for i in range(len(X) - ts):
        Xs.append(X[i:i + ts])
        ys.append(y[i + ts])
    return np.array(Xs), np.array(ys)

X_tr_seq, y_tr_seq = make_sequences(X_train, y_train, TIMESTEPS)
X_te_seq, y_te_seq = make_sequences(X_test,  y_test,  TIMESTEPS)
n_feat = X_tr_seq.shape[2]
log.info(f"Seq train: {X_tr_seq.shape}  |  Seq test: {X_te_seq.shape}")

# ── 7. Tree Models ─────────────────────────────────────────────────────────────
log.info("STEP 7: Training Tree Models (XGBoost + LightGBM)")

xgb_model = xgb.XGBRegressor(
    n_estimators=800, learning_rate=0.05, max_depth=7,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
    tree_method="hist", n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train_flat, y_train_flat, eval_set=[(X_test_flat, y_test_flat)], verbose=False)
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
log.info("XGBoost trained & saved.")

lgb_model = lgb.LGBMRegressor(
    n_estimators=800, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
    n_jobs=-1, verbosity=-1
)
lgb_model.fit(
    X_train_flat, y_train_flat,
    eval_set=[(X_test_flat, y_test_flat)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
joblib.dump(lgb_model, os.path.join(OUTPUT_DIR, "lgb_model.pkl"))
log.info("LightGBM trained & saved.")

# ── 8. SHAP Feature Ranking ────────────────────────────────────────────────────
log.info("STEP 8: SHAP Feature Ranking (LightGBM)")

shap_df = compute_shap_ranking(
    model        = lgb_model,
    X            = X_test_flat[:500],
    feature_names= feature_cols_final,
    output_dir   = OUTPUT_DIR,
    top_n        = 30,
    save_plot    = True,
)
if not shap_df.empty:
    log.info(f"Top 5 features by SHAP:\n{shap_df.head(5).to_string(index=False)}")

# ── 9. Metrics Helper ──────────────────────────────────────────────────────────
def metrics(y_true, y_pred, name):
    """Compute regression + directional accuracy metrics."""
    mse  = float(np.mean((y_true - y_pred)**2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    ss_r = np.sum((y_true - y_pred)**2)
    ss_t = np.sum((y_true - np.mean(y_true))**2)
    r2   = float(1 - ss_r / (ss_t + 1e-10))
    da   = float(
        np.mean(np.sign(np.diff(y_true.flatten())) == np.sign(np.diff(y_pred.flatten()))) * 100
    ) if len(y_true) > 1 else 0.0
    return {"Model": name, "MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "DA": da}

def inv(arr):
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

# ── 10. Deep Learning Models ───────────────────────────────────────────────────
log.info("STEP 10: Building & Training Deep Learning Models")

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, GRU, LSTM, Bidirectional,
    MultiHeadAttention, LayerNormalization, Dropout,
    GlobalAveragePooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import Huber

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    [tf.config.experimental.set_memory_growth(d, True) for d in gpus]
log.info(f"TF devices: {tf.config.list_physical_devices()}")

def get_callbacks(name):
    ckpt = os.path.join(OUTPUT_DIR, f"{name}_best.keras")
    return [
        EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(ckpt, save_best_only=True, monitor="val_loss", verbose=0),
    ]

def build_gru():
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = GRU(128, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = GRU(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, "relu")(x)
    m = Model(inp, Dense(1)(x))
    m.compile("adam", Huber(), metrics=["mae"])
    return m

def build_lstm():
    m = Sequential([
        LSTM(128, input_shape=(TIMESTEPS, n_feat), return_sequences=True),
        Dropout(0.2), LSTM(64), Dropout(0.2),
        Dense(32, "relu"), Dense(1),
    ])
    m.compile("adam", Huber(), metrics=["mae"])
    return m

class AttentionPool(tf.keras.layers.Layer):
    def __init__(self, u, **kw):
        super().__init__(**kw)
        self.q = Dense(u, activation="tanh")
        self.w = Dense(1)
    def call(self, x):
        return tf.reduce_sum(x * tf.nn.softmax(self.w(self.q(x)), axis=1), axis=1)

def build_bilstm():
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, "relu")(AttentionPool(64)(x))
    m = Model(inp, Dense(1)(x))
    m.compile("adam", Huber(), metrics=["mae"])
    return m

class TBlock(tf.keras.layers.Layer):
    def __init__(self, nh, dm, ff, dr=0.1, **kw):
        super().__init__(**kw)
        self.att = MultiHeadAttention(num_heads=nh, key_dim=dm // nh)
        self.ffn = Sequential([Dense(ff, "relu"), Dense(dm)])
        self.ln1 = LayerNormalization(1e-6)
        self.ln2 = LayerNormalization(1e-6)
        self.d1  = Dropout(dr)
        self.d2  = Dropout(dr)
    def call(self, x, training=False):
        x = self.ln1(x + self.d1(self.att(x, x), training=training))
        return self.ln2(x + self.d2(self.ffn(x), training=training))

def build_transformer():
    dm = min(n_feat, 64); nh = max(1, dm // 16)
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = TBlock(nh, dm, dm * 2)(Dense(dm)(inp))
    x = TBlock(nh, dm, dm * 2)(x)
    x = Dense(64, "relu")(GlobalAveragePooling1D()(x))
    m = Model(inp, Dense(1)(Dropout(0.2)(x)))
    m.compile("adam", Huber(), metrics=["mae"])
    return m

class TFTBlock(tf.keras.layers.Layer):
    def __init__(self, nh, dm, ff, dr=0.1, **kw):
        super().__init__(**kw)
        self.att  = MultiHeadAttention(num_heads=nh, key_dim=dm // nh)
        self.gate = Dense(dm, "sigmoid")
        self.ffn  = Sequential([Dense(ff, "relu"), Dense(dm)])
        self.ln1  = LayerNormalization(1e-6)
        self.ln2  = LayerNormalization(1e-6)
        self.d1   = Dropout(dr)
        self.d2   = Dropout(dr)
    def call(self, x, training=False):
        x = self.ln1(x + self.d1(self.att(x, x), training=training) * self.gate(x))
        return self.ln2(x + self.d2(self.ffn(x), training=training))

def build_tft():
    dm = min(n_feat, 64); nh = max(1, dm // 16)
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = TFTBlock(nh, dm, dm * 2)(Dense(dm)(inp))
    x = TFTBlock(nh, dm, dm * 2)(x)
    x = Dense(64, "relu")(GlobalAveragePooling1D()(x))
    m = Model(inp, Dense(1)(Dropout(0.2)(x)))
    m.compile("adam", Huber(), metrics=["mae"])
    return m

dl_builders = {
    "GRU": build_gru, "LSTM": build_lstm, "BiLSTM-Attn": build_bilstm,
    "Transformer": build_transformer, "TFT": build_tft,
}

results, histories, preds, dl_test_preds = [], {}, {}, {}

for name, fn in dl_builders.items():
    log.info(f"Training {name}...")
    m = fn()
    h = m.fit(
        X_tr_seq, y_tr_seq,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2, callbacks=get_callbacks(name), verbose=0,
    )
    p_sc = m.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0).flatten()
    dl_test_preds[name] = p_sc
    y_pr = inv(p_sc)
    y_tr = inv(y_te_seq.flatten())
    m_dict = metrics(y_tr, y_pr, name)
    results.append(m_dict)
    histories[name] = h
    preds[name] = (y_pr, y_tr)
    m.save(os.path.join(OUTPUT_DIR, f"{name.lower().replace('-','_').replace(' ','_')}.keras"))
    log.info(f"{name}: MAE={m_dict['MAE']:.4f}  R²={m_dict['R2']:.4f}  DA={m_dict['DA']:.1f}%")

# ── 11. Tree Model Evaluation ──────────────────────────────────────────────────
n_seq     = len(y_te_seq)
X_te_algn = X_test_flat[-n_seq:]
y_te_algn = y_test_flat[-n_seq:]

for tname, tmodel in [("XGBoost", xgb_model), ("LightGBM", lgb_model)]:
    p_sc   = tmodel.predict(X_te_algn).astype(np.float32)
    y_pr   = inv(p_sc)
    y_tr   = inv(y_te_algn)
    m_dict = metrics(y_tr, y_pr, tname)
    results.append(m_dict)
    preds[tname] = (y_pr, y_tr)
    log.info(f"{tname}: MAE={m_dict['MAE']:.4f}  R²={m_dict['R2']:.4f}  DA={m_dict['DA']:.1f}%")

# ── 12. Stacking Ensemble ──────────────────────────────────────────────────────
log.info("STEP 12: Stacking Ensemble")

meta_X = np.column_stack([
    *[dl_test_preds[n] for n in dl_builders],
    xgb_model.predict(X_te_algn),
    lgb_model.predict(X_te_algn),
])
meta_y   = y_te_seq.flatten()
mid      = int(len(meta_X) * 0.5)
meta_lr  = Ridge(alpha=1.0)
meta_lr.fit(meta_X[:mid], meta_y[:mid])
joblib.dump(meta_lr, os.path.join(OUTPUT_DIR, "stacking_meta.pkl"))

stack_pred = meta_lr.predict(meta_X[mid:])
y_pr_stack = inv(stack_pred)
y_tr_stack = inv(meta_y[mid:])
st_metrics = metrics(y_tr_stack, y_pr_stack, "Stacking Ensemble")
results.append(st_metrics)
preds["Stacking Ensemble"] = (y_pr_stack, y_tr_stack)
log.info(f"Stacking: MAE={st_metrics['MAE']:.4f}  R²={st_metrics['R2']:.4f}  DA={st_metrics['DA']:.1f}%")

# ── 13. Results & Visualizations ───────────────────────────────────────────────
log.info("STEP 13: Results & Visualizations")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
log.info("\n" + results_df.to_string(index=False))

# Loss curves
fig, axes = plt.subplots(1, len(dl_builders), figsize=(5 * len(dl_builders), 4), sharey=True)
for i, (nm, hist) in enumerate(histories.items()):
    ax = axes[i] if len(dl_builders) > 1 else axes
    ax.plot(hist.history["loss"],     label="Train")
    ax.plot(hist.history["val_loss"], label="Val", linestyle="--")
    ax.set_title(nm, fontsize=10); ax.set_xlabel("Epoch")
    if i == 0: ax.set_ylabel("Huber Loss")
    ax.legend(fontsize=7)
plt.suptitle("Training & Validation Loss", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# Metrics heatmap
mp = results_df.set_index("Model")[["MAE","RMSE","MAPE","R2","DA"]].astype(float)
np_norm = (mp - mp.min()) / (mp.max() - mp.min() + 1e-10)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(np_norm, annot=mp.round(4), fmt="", cmap="RdYlGn", ax=ax,
            linewidths=0.5, cbar_kws={"label": "Normalized Score"})
ax.set_title("Model Metrics Heatmap", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_heatmap.png"), dpi=150)
plt.close()

# Actual vs Predicted
for nm, (yp, yt) in preds.items():
    n = min(300, len(yt))
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(yt[:n], label="Actual", lw=1.5)
    ax.plot(yp[:n], label="Predicted", lw=1.0, alpha=0.85, linestyle="--")
    ax.set_title(f"{nm} – Actual vs Predicted", fontsize=11)
    ax.set_xlabel("Sample"); ax.set_ylabel("Exchange Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"pred_{nm.lower().replace(' ','_').replace('-','_')}.png"), dpi=150)
    plt.close()

log.info("All plots saved.")

best = min(results, key=lambda r: r["MAE"])
log.info("=" * 70)
log.info(f"BEST MODEL: {best['Model']}")
for k in ["MSE", "MAE", "RMSE", "MAPE", "R2", "DA"]:
    log.info(f"  {k:5s}: {best[k]:.6f}")
log.info("=" * 70)
log.info(f"Outputs → {os.path.abspath(OUTPUT_DIR)}")
log.info("Forex Prediction v3.1 complete ✓")
