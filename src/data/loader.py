"""
src/data/loader.py
==================
Data loading utilities for the Forex Ensemble Prediction pipeline.
"""

from __future__ import annotations
import logging
import pandas as pd

log = logging.getLogger(__name__)


def load_forex_data(path: str) -> pd.DataFrame:
    """
    Load and parse the Forex CSV dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file (e.g. 'Forex_Data.csv').

    Returns
    -------
    pd.DataFrame with columns: date (datetime), currency_code, exchange_rate.

    Raises
    ------
    FileNotFoundError if the path does not exist.
    ValueError if required columns are missing.
    """
    log.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df = df.drop(columns=["currency"], errors="ignore").dropna()

    required = {"date", "currency_code", "exchange_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    log.info(f"Loaded {len(df):,} rows, {df['currency_code'].nunique()} currencies.")
    return df
