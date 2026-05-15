"""
src/data/cleaner.py
===================
Data cleaning utilities: per-currency IQR outlier removal.
"""

from __future__ import annotations
import logging
import pandas as pd

log = logging.getLogger(__name__)


def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from exchange_rate using per-currency IQR filtering.

    For each currency_code group, values outside
    [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are dropped.

    Parameters
    ----------
    df         : DataFrame with columns [currency_code, exchange_rate].
    multiplier : IQR fence multiplier (default 1.5 = Tukey's rule).

    Returns
    -------
    Cleaned DataFrame, reset index.
    """
    cleaned = []
    for code, grp in df.groupby("currency_code"):
        q1, q3 = grp["exchange_rate"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = grp["exchange_rate"].between(lo, hi)
        dropped = (~mask).sum()
        if dropped:
            log.debug(f"{code}: removed {dropped} outliers.")
        cleaned.append(grp[mask])

    result = pd.concat(cleaned, ignore_index=True)
    result = result.sort_values(["currency_code", "date"]).reset_index(drop=True)
    log.info(f"After IQR outlier removal: {result.shape}")
    return result
