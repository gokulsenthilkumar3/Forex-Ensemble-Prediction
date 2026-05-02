"""
Multi-Factor Data Downloader
==============================
Downloads forex pairs + commodity cross-factors from Yahoo Finance.
Based on yahoo.txt ticker universe.
"""

import yfinance as yf
import pandas as pd
import os
import sys

os.makedirs("mf_data", exist_ok=True)

# ── Ticker Universe (from yahoo.txt) ──────────────────────────────────────
forex_pairs = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'NZDUSD=X', 'USDINR=X', 'USDCNY=X', 'USDSGD=X',
    'USDHKD=X', 'USDKRW=X', 'USDBRL=X', 'USDMXN=X', 'USDTRY=X',
    'USDZAR=X', 'USDNOK=X', 'USDSEK=X', 'USDDKK=X', 'USDPLN=X',
    'USDCZK=X', 'USDHUF=X', 'EURJPY=X', 'EURGBP=X', 'GBPJPY=X',
    'AUDJPY=X', 'CADJPY=X', 'CHFJPY=X', 'EURAUD=X', 'EURCAD=X'
]

commodities = [
    'GC=F', 'SI=F', 'PL=F', 'HG=F', 'CL=F', 'BZ=F', 'NG=F',
    'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'SB=F', 'CC=F', 'CT=F',
    'LE=F', 'HE=F', 'PA=F', 'RB=F', 'HO=F',
]

# Macro proxies – VIX, DXY, bond yields, equity indices
macro_tickers = [
    '^VIX',       # Volatility Index
    'DX-Y.NYB',   # US Dollar Index
    '^TNX',       # 10-Year Treasury Yield
    '^IRX',       # 13-Week T-Bill
    '^GSPC',      # S&P 500
    '^DJI',       # Dow Jones
    '^IXIC',      # NASDAQ
    '^FTSE',      # FTSE 100
    '^N225',      # Nikkei 225
]

ticker_labels = {
    'EURUSD=X':'EUR_USD','GBPUSD=X':'GBP_USD','USDJPY=X':'USD_JPY','AUDUSD=X':'AUD_USD',
    'USDCAD=X':'USD_CAD','USDCHF=X':'USD_CHF','NZDUSD=X':'NZD_USD','USDINR=X':'USD_INR',
    'USDCNY=X':'USD_CNY','USDSGD=X':'USD_SGD','USDHKD=X':'USD_HKD','USDKRW=X':'USD_KRW',
    'USDBRL=X':'USD_BRL','USDMXN=X':'USD_MXN','USDTRY=X':'USD_TRY','USDZAR=X':'USD_ZAR',
    'USDNOK=X':'USD_NOK','USDSEK=X':'USD_SEK','USDDKK=X':'USD_DKK','USDPLN=X':'USD_PLN',
    'USDCZK=X':'USD_CZK','USDHUF=X':'USD_HUF','EURJPY=X':'EUR_JPY','EURGBP=X':'EUR_GBP',
    'GBPJPY=X':'GBP_JPY','AUDJPY=X':'AUD_JPY','CADJPY=X':'CAD_JPY','CHFJPY=X':'CHF_JPY',
    'EURAUD=X':'EUR_AUD','EURCAD=X':'EUR_CAD',
    'GC=F':'Gold','SI=F':'Silver','PL=F':'Platinum','HG=F':'Copper',
    'CL=F':'WTI_Crude','BZ=F':'Brent_Crude','NG=F':'Natural_Gas',
    'ZC=F':'Corn','ZS=F':'Soybeans','ZW=F':'Wheat',
    'KC=F':'Coffee','SB=F':'Sugar','CC=F':'Cocoa','CT=F':'Cotton',
    'LE=F':'Live_Cattle','HE=F':'Lean_Hogs','PA=F':'Palladium',
    'RB=F':'RBOB_Gasoline','HO=F':'Heating_Oil',
    '^VIX':'VIX','^TNX':'US10Y','^IRX':'US3M',
    'DX-Y.NYB':'DXY','^GSPC':'SP500','^DJI':'DowJones',
    '^IXIC':'NASDAQ','^FTSE':'FTSE100','^N225':'Nikkei225',
}

all_tickers = forex_pairs + commodities + macro_tickers
all_dfs = []

print("=" * 60)
print("Downloading Multi-Factor Dataset from Yahoo Finance")
print("=" * 60)

for ticker in all_tickers:
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-05-01',
                         interval='1d', auto_adjust=True, progress=False)
        if df.empty:
            print(f"  ✗ {ticker}: empty")
            continue
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].copy()
        label = ticker_labels.get(ticker, ticker)
        df.columns = [f'{label}_{c}' for c in df.columns]
        all_dfs.append(df)
        print(f"  ✓ {label:20s}  rows={len(df)}")
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

if not all_dfs:
    print("ERROR: No data downloaded. Check internet connection.")
    sys.exit(1)

combined = pd.concat(all_dfs, axis=1)
combined.index.name = 'Date'
combined.sort_index(inplace=True)

# Forward-fill then backward-fill small gaps (weekends/holidays differ across markets)
combined = combined.ffill().bfill()

# Save
combined.to_csv('mf_data/multi_factor_ohlcv.csv')

close_cols = [c for c in combined.columns if c.endswith('_Close')]
combined[close_cols].to_csv('mf_data/multi_factor_close_only.csv')

print(f"\n✅ Saved!  Shape: {combined.shape}")
print(f"   Columns: {len(combined.columns)}  |  Rows: {len(combined)}")
print(f"   Date range: {combined.index.min()} → {combined.index.max()}")
print(f"   Files: mf_data/multi_factor_ohlcv.csv")
print(f"          mf_data/multi_factor_close_only.csv")
