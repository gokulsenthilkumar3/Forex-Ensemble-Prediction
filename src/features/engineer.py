"""
src/features/engineer.py
========================
Per-currency feature engineering with config-driven toggles.
All features respect the config/features.yaml settings.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
import os
import logging

log = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "features.yaml"
)


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    """Load feature configuration from a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_features(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add technical indicators and calendar features to a single-currency DataFrame.

    Parameters
    ----------
    g   : DataFrame for one currency_code, sorted by date.
    cfg : Loaded features.yaml config dict.

    Returns
    -------
    DataFrame with new feature columns appended.
    """
    g = g.copy()
    p = g["exchange_rate"]

    # ── Moving Averages ──────────────────────────────────────────────────────
    if cfg["moving_averages"]["enabled"]:
        for w in cfg["moving_averages"]["sma_windows"]:
            g[f"SMA_{w}"] = p.rolling(w, min_periods=1).mean()
        for w in cfg["moving_averages"]["ema_windows"]:
            g[f"EMA_{w}"] = p.ewm(span=w, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────────────────────
    if cfg["macd"]["enabled"]:
        fast, slow, sig = cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"]
        ema_fast = p.ewm(span=fast, adjust=False).mean()
        ema_slow = p.ewm(span=slow, adjust=False).mean()
        g["MACD"]        = ema_fast - ema_slow
        g["MACD_signal"] = g["MACD"].ewm(span=sig, adjust=False).mean()
        g["MACD_hist"]   = g["MACD"] - g["MACD_signal"]

    # ── RSI ──────────────────────────────────────────────────────────────────
    if cfg["rsi"]["enabled"]:
        period = cfg["rsi"]["period"]
        delta  = p.diff()
        gain   = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss   = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        g["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # ── Stochastic RSI ───────────────────────────────────────────────────────
    if cfg["stoch_rsi"]["enabled"] and "RSI" in g.columns:
        period  = cfg["stoch_rsi"]["period"]
        rsi     = g["RSI"]
        rsi_min = rsi.rolling(period, min_periods=1).min()
        rsi_max = rsi.rolling(period, min_periods=1).max()
        g["StochRSI"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    # ── Williams %R ──────────────────────────────────────────────────────────
    if cfg["williams_r"]["enabled"]:
        period = cfg["williams_r"]["period"]
        high   = p.rolling(period, min_periods=1).max()
        low    = p.rolling(period, min_periods=1).min()
        g["WilliamsR"] = -100 * (high - p) / (high - low + 1e-10)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if cfg["bollinger_bands"]["enabled"]:
        win    = cfg["bollinger_bands"]["window"]
        k      = cfg["bollinger_bands"]["std_dev"]
        sma    = p.rolling(win, min_periods=1).mean()
        std    = p.rolling(win, min_periods=1).std().fillna(0)
        g["BB_upper"] = sma + k * std
        g["BB_lower"] = sma - k * std
        g["BB_width"] = g["BB_upper"] - g["BB_lower"]
        g["BB_pct"]   = (p - g["BB_lower"]) / (g["BB_width"] + 1e-10)

    # ── ATR (only when OHLCV data is present) ────────────────────────────────
    if cfg["atr"]["enabled"] and {"high", "low", "close"}.issubset(g.columns):
        tr = pd.concat([
            g["high"] - g["low"],
            (g["high"] - g["close"].shift()).abs(),
            (g["low"]  - g["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        g["ATR"]     = tr.rolling(14, min_periods=1).mean()
        g["ATR_pct"] = g["ATR"] / (p + 1e-10)
    elif cfg["atr"]["enabled"]:
        log.debug("ATR skipped: high/low/close columns not found in dataset.")

    # ── OBV (only when volume is present) ────────────────────────────────────
    if cfg["obv"]["enabled"] and "volume" in g.columns:
        direction = np.sign(p.diff().fillna(0))
        g["OBV"] = (direction * g["volume"]).cumsum()
    elif cfg["obv"]["enabled"]:
        log.debug("OBV skipped: volume column not found in dataset.")

    # ── Returns & Volatility ─────────────────────────────────────────────────
    if cfg["returns_volatility"]["enabled"]:
        g["log_return"] = np.log(p / p.shift(1)).fillna(0)
        g["pct_change"] = p.pct_change().fillna(0)
        for w in cfg["returns_volatility"]["vol_windows"]:
            g[f"volatility_{w}"] = g["log_return"].rolling(w, min_periods=1).std().fillna(0)
        vols = cfg["returns_volatility"]["vol_windows"]
        if len(vols) >= 2:
            g["vol_ratio"] = g[f"volatility_{vols[0]}"] / (g[f"volatility_{vols[1]}"] + 1e-10)

    # ── Lag Features (ffill to prevent leakage) ───────────────────────────────
    if cfg["lags"]["enabled"]:
        fill = cfg["lags"].get("fill_method", "ffill")
        for lag in cfg["lags"]["periods"]:
            shifted = p.shift(lag)
            g[f"lag_{lag}"] = shifted.ffill() if fill == "ffill" else shifted.bfill()

    # ── Calendar Features ────────────────────────────────────────────────────
    if cfg["calendar"]["enabled"]:
        feats = cfg["calendar"]["features"]
        if "day_of_week"  in feats: g["day_of_week"]  = g["date"].dt.dayofweek
        if "month"        in feats: g["month"]        = g["date"].dt.month
        if "quarter"      in feats: g["quarter"]      = g["date"].dt.quarter
        if "is_month_end" in feats: g["is_month_end"] = g["date"].dt.is_month_end.astype(int)

    return g


def build_per_currency_scalers(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Fit a separate RobustScaler per currency_code to prevent high-rate
    currencies (e.g. JPY) from dominating the global feature scale.

    Returns
    -------
    dict mapping currency prefix patterns -> fitted scaler
    (For multi-currency one-hot encoded data, we fit a global scaler
    on each original currency slice before encoding.)
    """
    from sklearn.preprocessing import RobustScaler
    scalers = {}
    # This is called before one-hot encoding, on the raw grouped df
    for code, grp in df.groupby("currency_code"):
        sc = RobustScaler()
        sc.fit(grp[feature_cols].values)
        scalers[code] = sc
        log.debug(f"Fitted scaler for {code}")
    return scalers
