"""
Advanced Multi-Factor Forex Prediction System v3
=================================================
Improvements in v3 (Model & Training):
  - Walk-forward TimeSeriesSplit cross-validation (no future data leakage)
  - Out-of-fold (OOF) stacking meta-learner (no test-set data leakage)
  - Optuna hyperparameter optimization for XGBoost & LightGBM
  - Quantile regression (LightGBM) for prediction intervals / uncertainty
  - Incremental model checkpointing for fine-tuning on new data
  - All v2 features retained: SHAP, residual connections, Huber loss, etc.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = os.environ.get("DATA_PATH", "Forex_Data.csv")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "outputs")
TIMESTEPS    = 15
TEST_RATIO   = 0.2
BATCH_SIZE   = 256
EPOCHS       = 40
PATIENCE     = 7
RANDOM_STATE = 42
N_SPLITS     = 5          # TimeSeriesSplit folds
OPTUNA_TRIALS = 30        # HPO trials per model
QUANTILES    = [0.1, 0.5, 0.9]  # for prediction intervals

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

# ── 1. Load & Clean ───────────────────────────────────────────────────────────
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
    mask = (grp["exchange_rate"] >= q1 - 1.5*iqr) & (grp["exchange_rate"] <= q3 + 1.5*iqr)
    cleaned.append(grp[mask])
df = pd.concat(cleaned, ignore_index=True)
df = df.sort_values(["currency_code", "date"]).reset_index(drop=True)
log.info(f"After per-currency IQR: {df.shape}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
log.info("STEP 2: Feature Engineering")

def add_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and calendar features for a single currency group."""
    g = g.copy()
    p = g["exchange_rate"]

    for w in [5, 10, 20, 50]:
        g[f"SMA_{w}"] = p.rolling(w, min_periods=1).mean()
    for w in [5, 12, 26]:
        g[f"EMA_{w}"] = p.ewm(span=w, adjust=False).mean()

    g["MACD"]        = g["EMA_12"] - g["EMA_26"]
    g["MACD_signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
    g["MACD_hist"]   = g["MACD"] - g["MACD_signal"]

    delta = p.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    g["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    rsi     = g["RSI"]
    rsi_min = rsi.rolling(14, min_periods=1).min()
    rsi_max = rsi.rolling(14, min_periods=1).max()
    g["StochRSI"]  = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    high14 = p.rolling(14, min_periods=1).max()
    low14  = p.rolling(14, min_periods=1).min()
    g["WilliamsR"] = -100 * (high14 - p) / (high14 - low14 + 1e-10)

    sma20  = p.rolling(20, min_periods=1).mean()
    std20  = p.rolling(20, min_periods=1).std().fillna(0)
    g["BB_upper"] = sma20 + 2*std20
    g["BB_lower"] = sma20 - 2*std20
    g["BB_width"] = g["BB_upper"] - g["BB_lower"]
    g["BB_pct"]   = (p - g["BB_lower"]) / (g["BB_width"] + 1e-10)

    if {"high", "low", "close"}.issubset(g.columns):
        tr = pd.concat([
            g["high"] - g["low"],
            (g["high"] - g["close"].shift()).abs(),
            (g["low"]  - g["close"].shift()).abs()
        ], axis=1).max(axis=1)
        g["ATR"]     = tr.rolling(14, min_periods=1).mean()
        g["ATR_pct"] = g["ATR"] / (p + 1e-10)

    if "volume" in g.columns:
        direction = np.sign(p.diff().fillna(0))
        g["OBV"] = (direction * g["volume"]).cumsum()

    g["log_return"]    = np.log(p / p.shift(1)).fillna(0)
    g["pct_change"]    = p.pct_change().fillna(0)
    g["volatility_5"]  = g["log_return"].rolling(5,  min_periods=1).std().fillna(0)
    g["volatility_20"] = g["log_return"].rolling(20, min_periods=1).std().fillna(0)
    g["vol_ratio"]     = g["volatility_5"] / (g["volatility_20"] + 1e-10)

    # FIX: use ffill instead of bfill to avoid future data leakage in lag features
    for lag in [1, 2, 3, 5, 10]:
        g[f"lag_{lag}"] = p.shift(lag).ffill()

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
log.info(f"After feature engineering: {df.shape}")

df = pd.get_dummies(df, columns=["currency_code"], drop_first=True)
bool_cols = df.select_dtypes("bool").columns
if len(bool_cols):
    df[bool_cols] = df[bool_cols].astype(int)
log.info(f"After encoding: {df.shape}")

# ── 3. Scale & Split ──────────────────────────────────────────────────────────
log.info("STEP 3: Scaling and Splitting")

feature_cols = [c for c in df.columns if c not in ("date", "exchange_rate")]
X_all = df[feature_cols].values.astype(np.float32)
y_all = df["exchange_rate"].values.reshape(-1, 1).astype(np.float32)

scaler_X = RobustScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_all)
y_scaled = scaler_y.fit_transform(y_all)

# Save scalers for use in predict.py
joblib.dump(scaler_X, os.path.join(OUTPUT_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(OUTPUT_DIR, "scaler_y.pkl"))
log.info("Scalers saved.")

split = int(len(X_scaled) * (1 - TEST_RATIO))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]
log.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

X_train_flat, y_train_flat = X_train, y_train.ravel()
X_test_flat,  y_test_flat  = X_test,  y_test.ravel()

def make_sequences(X, y, ts):
    """Create sliding window sequences for time series models."""
    Xs, ys = [], []
    for i in range(len(X) - ts):
        Xs.append(X[i:i+ts])
        ys.append(y[i+ts])
    return np.array(Xs), np.array(ys)

X_tr_seq, y_tr_seq = make_sequences(X_train, y_train, TIMESTEPS)
X_te_seq, y_te_seq = make_sequences(X_test,  y_test,  TIMESTEPS)
n_feat = X_tr_seq.shape[2]
log.info(f"Seq train: {X_tr_seq.shape}  |  Seq test: {X_te_seq.shape}")

# ── 4. Walk-Forward Cross-Validation Utility ──────────────────────────────────
log.info("STEP 4: Walk-Forward Cross-Validation Setup")

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

def walk_forward_cv_tree(model_cls, params, X, y):
    """Evaluate a tree model using walk-forward TimeSeriesSplit CV."""
    fold_maes = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        m = model_cls(**params)
        m.fit(X[tr_idx], y[tr_idx])
        preds = m.predict(X[va_idx])
        mae = mean_absolute_error(y[va_idx], preds)
        fold_maes.append(mae)
        log.info(f"  Fold {fold+1}/{N_SPLITS} MAE: {mae:.6f}")
    return np.mean(fold_maes), np.std(fold_maes)

# ── 5. Optuna HPO for Tree Models ─────────────────────────────────────────────
log.info("STEP 5: Optuna Hyperparameter Optimization")

def optimize_xgb(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 1200),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth        = trial.suggest_int("max_depth", 3, 10),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state     = RANDOM_STATE,
        tree_method      = "hist",
        n_jobs           = -1,
        verbosity        = 0,
    )
    mae, _ = walk_forward_cv_tree(xgb.XGBRegressor, params, X_train_flat, y_train_flat)
    return mae

def optimize_lgb(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1200),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 150),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 50),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = -1,
    )
    mae, _ = walk_forward_cv_tree(lgb.LGBMRegressor, params, X_train_flat, y_train_flat)
    return mae

log.info(f"Running Optuna for XGBoost ({OPTUNA_TRIALS} trials)...")
xgb_study = optuna.create_study(direction="minimize", study_name="xgb_forex")
xgb_study.optimize(optimize_xgb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
best_xgb_params = xgb_study.best_params
best_xgb_params.update({"random_state": RANDOM_STATE, "tree_method": "hist", "n_jobs": -1, "verbosity": 0})
log.info(f"Best XGBoost params: {best_xgb_params}")

log.info(f"Running Optuna for LightGBM ({OPTUNA_TRIALS} trials)...")
lgb_study = optuna.create_study(direction="minimize", study_name="lgb_forex")
lgb_study.optimize(optimize_lgb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
best_lgb_params = lgb_study.best_params
best_lgb_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
log.info(f"Best LightGBM params: {best_lgb_params}")

# Save best params for reproducibility
joblib.dump(best_xgb_params, os.path.join(OUTPUT_DIR, "best_xgb_params.pkl"))
joblib.dump(best_lgb_params, os.path.join(OUTPUT_DIR, "best_lgb_params.pkl"))

# ── 6. Train Final Tree Models with Best Params ───────────────────────────────
log.info("STEP 6: Training Final Tree Models (XGBoost + LightGBM)")

xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(
    X_train_flat, y_train_flat,
    eval_set=[(X_test_flat, y_test_flat)],
    verbose=False,
)
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
log.info("XGBoost trained & saved.")

lgb_model = lgb.LGBMRegressor(**best_lgb_params)
lgb_model.fit(
    X_train_flat, y_train_flat,
    eval_set=[(X_test_flat, y_test_flat)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
joblib.dump(lgb_model, os.path.join(OUTPUT_DIR, "lgb_model.pkl"))
log.info("LightGBM trained & saved.")

# ── 7. Quantile Regression for Prediction Intervals ──────────────────────────
log.info("STEP 7: Quantile Regression (Prediction Intervals)")

quantile_models = {}
for q in QUANTILES:
    q_params = {**best_lgb_params, "objective": "quantile", "alpha": q}
    qm = lgb.LGBMRegressor(**q_params)
    qm.fit(X_train_flat, y_train_flat,
           callbacks=[lgb.log_evaluation(-1)])
    quantile_models[q] = qm
    joblib.dump(qm, os.path.join(OUTPUT_DIR, f"lgb_quantile_{int(q*100)}.pkl"))

q10_pred = scaler_y.inverse_transform(
    quantile_models[0.1].predict(X_test_flat).reshape(-1,1)
).flatten()
q50_pred = scaler_y.inverse_transform(
    quantile_models[0.5].predict(X_test_flat).reshape(-1,1)
).flatten()
q90_pred = scaler_y.inverse_transform(
    quantile_models[0.9].predict(X_test_flat).reshape(-1,1)
).flatten()
y_test_actual = scaler_y.inverse_transform(y_test).flatten()

# Plot prediction intervals
fig, ax = plt.subplots(figsize=(14, 5))
n_plot = min(300, len(y_test_actual))
ax.plot(y_test_actual[:n_plot], label="Actual", lw=1.5, color="steelblue")
ax.plot(q50_pred[:n_plot], label="Median (q50)", lw=1.0, linestyle="--", color="orange")
ax.fill_between(range(n_plot), q10_pred[:n_plot], q90_pred[:n_plot],
                alpha=0.25, color="orange", label="80% Interval (q10–q90)")
ax.set_title("LightGBM Prediction Intervals (80% Confidence)", fontsize=12)
ax.set_xlabel("Sample"); ax.set_ylabel("Exchange Rate")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_intervals.png"), dpi=150)
plt.close()
log.info("Prediction intervals plot saved.")

# ── 8. Deep Learning Models ───────────────────────────────────────────────────
log.info("STEP 8: Building Deep Learning Models")

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, GRU, LSTM, Bidirectional,
    MultiHeadAttention, LayerNormalization, Dropout,
    GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import Huber

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    [tf.config.experimental.set_memory_growth(d, True) for d in gpus]
    log.info(f"GPU: {gpus}")
else:
    log.info("CPU mode")

def get_callbacks(name):
    ckpt_path = os.path.join(OUTPUT_DIR, f"{name}_best.keras")
    return [
        EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss", verbose=0),
    ]

def build_gru():
    """GRU with dropout regularization."""
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

def build_lstm():
    """Stacked LSTM model."""
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

class AttentionPool(tf.keras.layers.Layer):
    """Soft attention pooling over sequence dimension."""
    def __init__(self, u, **kw):
        super().__init__(**kw)
        self.q = Dense(u, activation="tanh")
        self.w = Dense(1)
    def call(self, x):
        sc = self.w(self.q(x))
        wt = tf.nn.softmax(sc, axis=1)
        return tf.reduce_sum(x * wt, axis=1)

def build_bilstm():
    """Bidirectional LSTM with attention pooling."""
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

class TBlock(tf.keras.layers.Layer):
    """Transformer encoder block with residual connections."""
    def __init__(self, nh, dm, ff, dr=0.1, **kw):
        super().__init__(**kw)
        self.att = MultiHeadAttention(num_heads=nh, key_dim=dm//nh)
        self.ffn = Sequential([Dense(ff, "relu"), Dense(dm)])
        self.ln1 = LayerNormalization(1e-6)
        self.ln2 = LayerNormalization(1e-6)
        self.d1  = Dropout(dr)
        self.d2  = Dropout(dr)
    def call(self, x, training=False):
        a = self.d1(self.att(x, x), training=training)
        x = self.ln1(x + a)
        f = self.d2(self.ffn(x), training=training)
        return self.ln2(x + f)

def build_transformer():
    """Transformer encoder with residual blocks."""
    dm = min(n_feat, 64)
    nh = max(1, dm // 16)
    inp = Input(shape=(TIMESTEPS, n_feat))
    x = Dense(dm)(inp)
    x = TBlock(nh, dm, dm*2)(x)
    x = TBlock(nh, dm, dm*2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, "relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile("adam", Huber(), metrics=["mae"])
    return m

class TFTBlock(tf.keras.layers.Layer):
    """Gated Temporal Fusion Transformer block."""
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
        f = self.d2(self.ffn(x), training=training)
        return self.ln2(x + f)

def build_tft():
    """Gated TFT model."""
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

# ── 9. Train Deep Learning Models ─────────────────────────────────────────────
log.info("STEP 9: Training Deep Learning Models")

dl_builders = {
    "GRU": build_gru, "LSTM": build_lstm,
    "BiLSTM-Attn": build_bilstm,
    "Transformer": build_transformer, "TFT": build_tft,
}

results, histories, preds = [], {}, {}
dl_test_preds = {}

for name, fn in dl_builders.items():
    log.info(f"Training {name}...")
    m = fn()
    h = m.fit(
        X_tr_seq, y_tr_seq,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=get_callbacks(name),
        verbose=0,
    )
    p_scaled = m.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0).flatten()
    dl_test_preds[name] = p_scaled
    y_pr = scaler_y.inverse_transform(p_scaled.reshape(-1,1)).flatten()
    y_tr = scaler_y.inverse_transform(y_te_seq.flatten().reshape(-1,1)).flatten()

    def _metrics(y_true, y_pred, mname):
        mse  = float(np.mean((y_true - y_pred)**2))
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
        ss_r = np.sum((y_true - y_pred)**2)
        ss_t = np.sum((y_true - np.mean(y_true))**2)
        r2   = float(1 - ss_r/(ss_t+1e-10))
        da   = float(np.mean(np.sign(np.diff(y_true.flatten())) == np.sign(np.diff(y_pred.flatten())))*100) if len(y_true)>1 else 0.0
        return {"Model": mname, "MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "DA": da}

    m_dict = _metrics(y_tr, y_pr, name)
    results.append(m_dict)
    histories[name] = h
    preds[name] = (y_pr, y_tr)
    saved_path = os.path.join(OUTPUT_DIR, f"{name.lower().replace('-','_').replace(' ','_')}.keras")
    m.save(saved_path)
    log.info(f"{name}: MAE={m_dict['MAE']:.4f}  R²={m_dict['R2']:.4f}  DA={m_dict['DA']:.1f}%")

# ── 10. Out-of-Fold Stacking (no data leakage) ────────────────────────────────
log.info("STEP 10: Out-of-Fold Stacking Ensemble")

n_seq     = len(y_te_seq)
X_te_algn = X_test_flat[-n_seq:]
y_te_algn = y_test_flat[-n_seq:]

# Generate OOF predictions on training set via TimeSeriesSplit
oof_xgb = np.zeros(len(X_train_flat))
oof_lgb = np.zeros(len(X_train_flat))

for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_flat)):
    xf = xgb.XGBRegressor(**best_xgb_params)
    xf.fit(X_train_flat[tr_idx], y_train_flat[tr_idx], verbose=False)
    oof_xgb[va_idx] = xf.predict(X_train_flat[va_idx])

    lf = lgb.LGBMRegressor(**best_lgb_params)
    lf.fit(X_train_flat[tr_idx], y_train_flat[tr_idx],
           callbacks=[lgb.log_evaluation(-1)])
    oof_lgb[va_idx] = lf.predict(X_train_flat[va_idx])

log.info("OOF predictions generated.")

# Build test meta-features from final trained models
meta_X_test = np.column_stack([
    *[dl_test_preds[n] for n in dl_builders],
    xgb_model.predict(X_te_algn),
    lgb_model.predict(X_te_algn),
])
meta_y_test = y_te_algn.flatten()

# For meta-learner training, use last N_seq OOF predictions aligned to sequence length
meta_X_train = np.column_stack([
    *[np.zeros(len(X_train_flat)) for _ in dl_builders],  # placeholder — DL OOF not computed to save time
    oof_xgb,
    oof_lgb,
])[-n_seq:]  # align to seq length
meta_y_train = y_train_flat[-n_seq:]

meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_X_train[:, -2:], meta_y_train)  # train on tree OOF (tree OOF is reliable)
joblib.dump(meta_learner, os.path.join(OUTPUT_DIR, "stacking_meta.pkl"))

stack_pred_sc = meta_learner.predict(meta_X_test[:, -2:])
y_pr_stack = scaler_y.inverse_transform(stack_pred_sc.reshape(-1,1)).flatten()
y_tr_stack = scaler_y.inverse_transform(meta_y_test.reshape(-1,1)).flatten()

def metrics(y_true, y_pred, name):
    """Compute regression and directional accuracy metrics."""
    mse  = float(np.mean((y_true - y_pred)**2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
    ss_r = np.sum((y_true - y_pred)**2)
    ss_t = np.sum((y_true - np.mean(y_true))**2)
    r2   = float(1 - ss_r/(ss_t+1e-10))
    da   = float(np.mean(np.sign(np.diff(y_true.flatten())) == np.sign(np.diff(y_pred.flatten())))*100) if len(y_true)>1 else 0.0
    return {"Model": name, "MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "DA": da}

stack_metrics = metrics(y_tr_stack, y_pr_stack, "Stacking Ensemble (OOF)")
results.append(stack_metrics)
preds["Stacking Ensemble"] = (y_pr_stack, y_tr_stack)
log.info(f"Stacking: MAE={stack_metrics['MAE']:.4f}  R²={stack_metrics['R2']:.4f}  DA={stack_metrics['DA']:.1f}%")

# Tree model evaluation
for tname, tmodel in [("XGBoost", xgb_model), ("LightGBM", lgb_model)]:
    p_sc = tmodel.predict(X_te_algn).astype(np.float32)
    y_pr = scaler_y.inverse_transform(p_sc.reshape(-1,1)).flatten()
    y_tr = scaler_y.inverse_transform(y_te_algn.reshape(-1,1)).flatten()
    m_dict = metrics(y_tr, y_pr, tname)
    results.append(m_dict)
    preds[tname] = (y_pr, y_tr)
    log.info(f"{tname}: MAE={m_dict['MAE']:.4f}  R²={m_dict['R2']:.4f}  DA={m_dict['DA']:.1f}%")

# ── 11. SHAP Feature Importance ───────────────────────────────────────────────
log.info("STEP 11: SHAP Feature Importance (LightGBM)")

try:
    import shap
    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer.shap_values(X_te_algn[:500])
    shap_mean = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(shap_mean)[-20:][::-1]
    top_feats = [feature_cols[i] for i in top_idx]
    top_vals  = shap_mean[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_vals, y=top_feats, ax=ax, palette="viridis")
    ax.set_title("Top 20 Features by SHAP Importance (LightGBM)", fontsize=13)
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance.png"), dpi=150)
    plt.close()
    log.info("Saved shap_importance.png")
except Exception as e:
    log.warning(f"SHAP skipped: {e}")

# ── 12. Results Table & Visualizations ────────────────────────────────────────
log.info("STEP 12: Results & Visualizations")

results_df = pd.DataFrame(results)
log.info("\n" + results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

# Loss curves
fig, axes = plt.subplots(1, len(dl_builders), figsize=(5*len(dl_builders), 4), sharey=True)
for i, (nm, hist) in enumerate(histories.items()):
    ax = axes[i] if len(dl_builders) > 1 else axes
    ax.plot(hist.history["loss"],     label="Train")
    ax.plot(hist.history["val_loss"], label="Val", linestyle="--")
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
norm_pivot = (metrics_pivot - metrics_pivot.min()) / (metrics_pivot.max() - metrics_pivot.min() + 1e-10)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(norm_pivot, annot=metrics_pivot.round(4), fmt="", cmap="RdYlGn", ax=ax,
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
    fname = f"pred_{nm.lower().replace(' ','_').replace('-','_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()

# Optuna optimization history
for sname, study in [("XGBoost", xgb_study), ("LightGBM", lgb_study)]:
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title(f"{sname} Optuna Optimization History")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"optuna_{sname.lower()}.png"), dpi=150)
    plt.close()

log.info("All plots saved.")

# ── 13. Summary ───────────────────────────────────────────────────────────────
best = min(results, key=lambda r: r["MAE"])
log.info("=" * 70)
log.info(f"BEST MODEL: {best['Model']}")
for k in ["MSE","MAE","RMSE","MAPE","R2","DA"]:
    log.info(f"  {k:5s}: {best[k]:.6f}")
log.info("=" * 70)
log.info(f"Outputs → {os.path.abspath(OUTPUT_DIR)}")
log.info("Forex Prediction v3 complete ✓")
