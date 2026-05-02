import pandas as pd
import numpy as np
from tqdm import tqdm

def add_technical_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """Adds technical indicators and lag features to a currency group."""
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

    # RSI
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    g["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # Stochastic RSI
    rsi = g["RSI"]
    rsi_min = rsi.rolling(14, min_periods=1).min()
    rsi_max = rsi.rolling(14, min_periods=1).max()
    g["StochRSI"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    # Bollinger Bands
    sma20 = p.rolling(20, min_periods=1).mean()
    std20 = p.rolling(20, min_periods=1).std().fillna(0)
    g["BB_upper"] = sma20 + 2*std20
    g["BB_lower"] = sma20 - 2*std20
    g["BB_width"] = g["BB_upper"] - g["BB_lower"]

    # Returns & Volatility
    g["log_return"] = np.log(p / p.shift(1)).fillna(0)
    g["volatility_20"] = g["log_return"].rolling(20, min_periods=1).std().fillna(0)

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        g[f"lag_{lag}"] = p.shift(lag).bfill()

    # Calendar features
    g["day_of_week"] = g["date"].dt.dayofweek
    g["month"] = g["date"].dt.month
    g["is_month_end"] = g["date"].dt.is_month_end.astype(int)

    return g

def build_feature_set(df):
    """Orchestrates feature engineering for all currencies."""
    dfs = []
    for code, grp in tqdm(df.groupby("currency_code"), desc="Engineering Features"):
        g = add_technical_indicators(grp)
        g["currency_code"] = code
        dfs.append(g)

    df_feats = pd.concat(dfs, ignore_index=True).dropna().reset_index(drop=True)
    
    # One-hot encode currency
    df_feats = pd.get_dummies(df_feats, columns=["currency_code"], drop_first=True)
    bool_cols = df_feats.select_dtypes("bool").columns
    if len(bool_cols):
        df_feats[bool_cols] = df_feats[bool_cols].astype(int)
        
    return df_feats
