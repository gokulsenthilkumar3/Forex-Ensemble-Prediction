import os
import json
import sqlite3
import asyncio
import aiohttp
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class AsyncForexDataDownloader:
    def __init__(self, config_path="config/tickers.json", api_key=None, db_path="mf_data_cache.db"):
        self.config_path = config_path
        self.api_key = api_key or os.getenv("OANDA_API_KEY", "YOUR_OANDA_API_KEY")
        self.db_path = db_path
        self.config = self._load_config()
        self.base_url = self.config.get("api_url", "https://api-fxpractice.oanda.com/v3")
        self.semaphore = asyncio.Semaphore(10)
        self._init_db()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}. Using defaults.")
            return {"forex_pairs": ["EUR_USD"], "interval": "H1", "history_days": 30}
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _init_db(self):
        """Initializes SQLite database to store historical OHLCV data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    instrument TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (instrument, timestamp)
                )
            ''')
            conn.commit()

    async def fetch_ohlcv_from_oanda(self, session, instrument, start_date, end_date, interval):
        """Fetches candlestick data asynchronously from OANDA API."""
        url = f"{self.base_url}/instruments/{instrument}/candles"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept-Datetime-Format": "UNIX"
        }
        
        params = {
            "granularity": interval,
            "from": str(int(start_date.timestamp())),
            "to": str(int(end_date.timestamp()))
        }
        
        logger.info(f"Fetching data for {instrument} from OANDA...")
        try:
            async with self.semaphore:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"API request failed for {instrument} [{response.status}]: {text}")
                        return pd.DataFrame()
                    
                    data = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Network error fetching {instrument}: {e}")
            return pd.DataFrame()
            
        candles = data.get("candles", [])
        if not candles:
            return pd.DataFrame()

        rows = []
        for c in candles:
            if not c.get("complete", False):
                continue
            rows.append({
                "timestamp": pd.to_datetime(float(c["time"]), unit="s"),
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": int(c["volume"])
            })
        
        df = pd.DataFrame(rows)
        # Insert into local cache database
        await asyncio.to_thread(self._cache_data, instrument, df)
        return df

    def _cache_data(self, instrument, df):
        """Writes raw Pandas DataFrame to SQLite using UPSERT (INSERT OR REPLACE)"""
        if df.empty:
            return
        with sqlite3.connect(self.db_path) as conn:
            df_to_sql = df.copy()
            df_to_sql["instrument"] = instrument
            
            cursor = conn.cursor()
            for _, row in df_to_sql.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO ohlcv_cache 
                    (instrument, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row["instrument"], 
                    row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), 
                    row["open"], row["high"], row["low"], row["close"], row["volume"]
                ))
            conn.commit()

    def get_cached_data(self, instrument, start_date, end_date):
        """Retrieves raw cached historical data for a specific instrument and time range."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_cache 
                WHERE instrument = ? 
                AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            '''
            df = pd.read_sql_query(
                query, conn, 
                params=(instrument, start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S")),
                parse_dates=["timestamp"]
            )
            return df

    def enrich_features(self, df):
        """
        Computes Forex characteristics and technical indicators 
        needed for the ensemble model on the raw dataframe.
        """
        if df.empty:
            return df
        
        data = df.copy()
        data.set_index("timestamp", inplace=True)

        data["returns"] = data["close"].pct_change()
        data["log_returns"] = np.log(data["close"] / data["close"].shift(1))
        
        # Moving Averages
        data["sma_20"] = data["close"].rolling(window=20).mean()
        data["sma_50"] = data["close"].rolling(window=50).mean()
        data["ema_12"] = data["close"].ewm(span=12, adjust=False).mean()
        data["ema_26"] = data["close"].ewm(span=26, adjust=False).mean()

        # MACD
        data["macd"] = data["ema_12"] - data["ema_26"]
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

        # Volatility & Bollinger Bands
        data["volatility_20"] = data["returns"].rolling(window=20).std()
        data["bollinger_high"] = data["sma_20"] + (data["volatility_20"] * 2)
        data["bollinger_low"] = data["sma_20"] - (data["volatility_20"] * 2)

        # RSI (14 periods)
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        data["rsi_14"] = 100 - (100 / (1 + rs))

        data.dropna(inplace=True)
        return data

    async def run_pipeline(self):
        end_date = datetime.utcnow()
        days = self.config.get("history_days", 30)
        start_date = end_date - timedelta(days=days)
        interval = self.config.get("interval", "H1")
        pairs = self.config.get("pairs", [])
        commodities = self.config.get("commodities", [])
        macro = self.config.get("macro_tickers", [])
        
        all_instruments = pairs + commodities + macro

        processed_datasets = {}

        # 1. Fetch missing/new data from API concurrently
        async with aiohttp.ClientSession() as session:
            tasks = []
            for instrument in all_instruments:
                tasks.append(self.fetch_ohlcv_from_oanda(session, instrument, start_date, end_date, interval))
            
            # Execute all network requests asynchronously
            await asyncio.gather(*tasks)

        # 2. Extract from local Cache and Enrich with Technical Indicators
        for instrument in all_instruments:
            logger.info(f"Enriching features for {instrument}...")
            cached_df = self.get_cached_data(instrument, start_date, end_date)
            enriched_df = self.enrich_features(cached_df)
            processed_datasets[instrument] = enriched_df
            if not enriched_df.empty:
                logger.info(f"{instrument} processed successfully (Rows ready for modeling: {len(enriched_df)}).")
            else:
                logger.warning(f"No data available for {instrument}.")

        return processed_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async OANDA Downloader & Feature Engineer")
    parser.add_argument("--config", default="config/tickers.json", help="Path to tickers config")
    parser.add_argument("--db", default="mf_data_cache.db", help="Path to SQLite cache DB")
    args = parser.parse_args()

    downloader = AsyncForexDataDownloader(config_path=args.config, db_path=args.db)
    
    # Run async pipeline
    loop = asyncio.get_event_loop()
    datasets = loop.run_until_complete(downloader.run_pipeline())
    
    logger.info(f"Pipeline finished processing {len(datasets)} pairs.")
