"""
tests/test_features.py
=======================
Unit tests for src/features/engineer.py
"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineer import add_features


DEFAULT_CFG = {
    "moving_averages":     {"enabled": True,  "sma_windows": [5, 10], "ema_windows": [5, 12, 26]},
    "macd":                {"enabled": True,  "fast": 12, "slow": 26, "signal": 9},
    "rsi":                 {"enabled": True,  "period": 14},
    "stoch_rsi":           {"enabled": True,  "period": 14},
    "williams_r":          {"enabled": True,  "period": 14},
    "bollinger_bands":     {"enabled": True,  "window": 20, "std_dev": 2},
    "atr":                 {"enabled": False},
    "obv":                 {"enabled": False},
    "returns_volatility":  {"enabled": True,  "vol_windows": [5, 20]},
    "lags":                {"enabled": True,  "periods": [1, 2, 3], "fill_method": "ffill"},
    "calendar":            {"enabled": True,  "features": ["day_of_week", "month", "quarter", "is_month_end"]},
}


def _make_group(n=60):
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "currency_code": "USD",
        "exchange_rate": np.linspace(80.0, 85.0, n) + np.random.randn(n) * 0.1,
    })


def test_feature_columns_added():
    df = _make_group()
    out = add_features(df, DEFAULT_CFG)
    expected = ["SMA_5", "EMA_12", "MACD", "RSI", "StochRSI", "WilliamsR",
                "BB_upper", "BB_pct", "log_return", "volatility_5",
                "lag_1", "lag_3", "day_of_week", "month", "is_month_end"]
    for col in expected:
        assert col in out.columns, f"Missing feature column: {col}"


def test_lag_uses_ffill_not_bfill():
    """lag_1 should be NaN at row 0 after ffill (not filled from the future)."""
    df = _make_group(10)
    out = add_features(df, DEFAULT_CFG)
    # After ffill, row 0 lag_1 should be NaN (nothing to fill forward from)
    assert pd.isna(out["lag_1"].iloc[0])


def test_atr_not_added_when_disabled():
    df = _make_group()
    out = add_features(df, DEFAULT_CFG)  # atr.enabled = False
    assert "ATR" not in out.columns


def test_no_future_leakage_in_lags():
    """Verify lag_k at row i equals exchange_rate at row i-k."""
    df = _make_group(20)
    out = add_features(df, DEFAULT_CFG)
    for k in [1, 2, 3]:
        for i in range(k + 1, len(out)):
            assert out[f"lag_{k}"].iloc[i] == pytest.approx(
                out["exchange_rate"].iloc[i - k], rel=1e-5
            )


def test_calendar_features_range():
    df = _make_group(60)
    out = add_features(df, DEFAULT_CFG)
    assert out["day_of_week"].between(0, 6).all()
    assert out["month"].between(1, 12).all()
    assert out["quarter"].between(1, 4).all()
    assert out["is_month_end"].isin([0, 1]).all()
