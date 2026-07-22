"""
src/data/api_fetcher.py
======================
Dynamic Live API data downloader for Forex pairs, Commodities, and Macro indicators.
Fetches 30+ years of historical data (1990 - Present) using Yahoo Finance API.
"""

from __future__ import annotations
import os
import logging
from datetime import datetime
import pandas as pd
import yfinance as yf

import json

log = logging.getLogger(__name__)

def load_tickers_config():
    paths = [
        "config/tickers.json",
        "../config/tickers.json",
        "Forex-Ensemble-Prediction/config/tickers.json"
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError("Could not find tickers.json in config folder.")

_TICKERS_CFG = load_tickers_config()
FOREX_PAIRS = _TICKERS_CFG["forex_pairs"]
COMMODITIES = _TICKERS_CFG["commodities"]
MACRO_TICKERS = _TICKERS_CFG["macro_tickers"]
TICKER_LABELS = _TICKERS_CFG["ticker_labels"]


def fetch_live_multi_factor_data(
    start_date: str = "1990-01-01",
    end_date: str | None = None,
    output_dir: str = "mf_data",
    update_forex_csv: bool = True,
) -> pd.DataFrame:
    """
    Downloads historical multi-factor forex + commodities + macro data from Yahoo Finance API.
    
    Parameters
    ----------
    start_date : str
        Earliest start date for historical fetch (e.g. '1990-01-01' for 30+ years).
    end_date : str | None
        End date for fetch (defaults to today's date).
    output_dir : str
        Directory to save output multi-factor CSV files.
    update_forex_csv : bool
        If True, formats forex pair data into `Forex_Data.csv` format and saves it.

    Returns
    -------
    pd.DataFrame containing merged OHLCV features across all tickers.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(output_dir, exist_ok=True)
    all_tickers = FOREX_PAIRS + COMMODITIES + MACRO_TICKERS
    all_dfs = []
    forex_long_records = []

    log.info(f"Downloading 30+ Year Multi-Factor Dataset ({start_date} -> {end_date})...")

    for ticker in all_tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                log.warning(f"  ✗ {ticker}: empty response")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            label = TICKER_LABELS.get(ticker, ticker)

            # Store long format for Forex pairs if requested
            if update_forex_csv and ticker in FOREX_PAIRS:
                curr_code = label.replace("_", "/")
                f_df = df.reset_index().copy()
                f_df["currency_code"] = curr_code
                f_df.rename(columns={
                    "Date": "date",
                    "Close": "exchange_rate",
                }, inplace=True)
                forex_long_records.append(f_df[["date", "exchange_rate", "High", "Low", "Open", "currency_code"]])

            df.columns = [f"{label}_{c}" for c in df.columns]
            all_dfs.append(df)
            log.info(f"  ✓ {label:20s} rows={len(df)}")
        except Exception as e:
            log.error(f"  ✗ {ticker}: {e}")

    if not all_dfs:
        raise ValueError("No data could be fetched from Yahoo Finance.")

    combined = pd.concat(all_dfs, axis=1)
    combined.index.name = "Date"
    combined.sort_index(inplace=True)
    combined = combined.ffill().bfill()

    # Save multi-factor CSVs
    ohlcv_path = os.path.join(output_dir, "multi_factor_ohlcv.csv")
    close_path = os.path.join(output_dir, "multi_factor_close_only.csv")
    combined.to_csv(ohlcv_path)

    close_cols = [c for c in combined.columns if c.endswith("_Close")]
    combined[close_cols].to_csv(close_path)

    log.info(f"Saved Multi-Factor data: {combined.shape} -> {ohlcv_path}")

    # Optionally create/update Forex_Data.csv with 30+ year forex data
    if update_forex_csv and forex_long_records:
        forex_all = pd.concat(forex_long_records, ignore_index=True)
        forex_all["date"] = pd.to_datetime(forex_all["date"]).dt.strftime("%d/%m/%Y")
        forex_all.to_csv("Forex_Data.csv", index=False)
        log.info(f"Updated Forex_Data.csv with {len(forex_all):,} historical records.")

    return combined
