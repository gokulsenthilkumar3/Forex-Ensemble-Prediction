"""
Multi-Factor Data Downloader
==============================
Downloads 30+ years of historical forex pairs + commodity cross-factors from Yahoo Finance API.
"""

import argparse
import logging
from datetime import datetime
from src.data.api_fetcher import fetch_live_multi_factor_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch 30+ years of Forex, Commodity, and Macro API data.")
    parser.add_argument("--start", default="1990-01-01", help="Start date (default: 1990-01-01 for 30+ years)")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (default: today)")
    parser.add_argument("--output-dir", default="mf_data", help="Output directory")
    args = parser.parse_args()

    fetch_live_multi_factor_data(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        update_forex_csv=True,
    )
