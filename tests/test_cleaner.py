"""
tests/test_cleaner.py
=====================
Unit tests for src/data/cleaner.py
"""

import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import remove_outliers_iqr


def _make_df(values, code="USD"):
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=len(values)),
        "currency_code": code,
        "exchange_rate": values,
    })


def test_removes_high_outlier():
    vals = list(range(1, 101)) + [10_000]   # 10_000 is a clear outlier
    df = _make_df(vals)
    clean = remove_outliers_iqr(df)
    assert 10_000 not in clean["exchange_rate"].values


def test_keeps_normal_values():
    vals = list(range(1, 51))
    df = _make_df(vals)
    clean = remove_outliers_iqr(df)
    # No outliers: all values should be retained
    assert len(clean) == len(df)


def test_per_currency_isolation():
    """Outlier in one currency should not affect another."""
    df_usd = _make_df(list(range(1, 51)), code="USD")
    df_eur = _make_df(list(range(1, 51)) + [99_999], code="EUR")
    df = pd.concat([df_usd, df_eur], ignore_index=True)
    clean = remove_outliers_iqr(df)
    # USD should still have 50 rows
    assert len(clean[clean["currency_code"] == "USD"]) == 50
    # EUR outlier should be removed
    assert 99_999 not in clean[clean["currency_code"] == "EUR"]["exchange_rate"].values


def test_output_sorted_by_currency_date():
    vals = list(range(10, 0, -1))  # reverse order
    df = _make_df(vals)
    clean = remove_outliers_iqr(df)
    assert list(clean["date"]) == sorted(clean["date"].tolist())
