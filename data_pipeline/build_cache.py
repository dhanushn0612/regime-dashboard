"""
Cache Builder — 20-Year Price History
=======================================
Downloads full price history for 48 core stocks going back to 2006.
Merges with existing Nifty 500 cache for the recent period.
Run this ONCE before the 20-year backtest.

Usage:
    python data_pipeline/build_cache.py
"""

import os
import pickle
import random
import time
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH  = os.path.join(SCRIPT_DIR, 'nifty500_prices.pkl')

UNIVERSE = [
    'HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','KOTAKBANK.NS','AXISBANK.NS',
    'TCS.NS','INFY.NS','HCLTECH.NS','WIPRO.NS','TECHM.NS',
    'HINDUNILVR.NS','ITC.NS','NESTLEIND.NS','BRITANNIA.NS','DABUR.NS',
    'SUNPHARMA.NS','DRREDDY.NS','CIPLA.NS','DIVISLAB.NS','AUROPHARMA.NS',
    'MARUTI.NS','BAJAJ-AUTO.NS','HEROMOTOCO.NS','EICHERMOT.NS',
    'RELIANCE.NS','ONGC.NS','COALINDIA.NS','TATASTEEL.NS','JSWSTEEL.NS',
    'LT.NS','NTPC.NS','POWERGRID.NS','TITAN.NS','ASIANPAINT.NS',
    'BAJFINANCE.NS','MUTHOOTFIN.NS','COFORGE.NS','PERSISTENT.NS',
    'PIDILITIND.NS','HAVELLS.NS','POLYCAB.NS','ABB.NS','SIEMENS.NS',
    'INDHOTEL.NS','DMART.NS','TRENT.NS','LTTS.NS','KPITTECH.NS',
]

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_close(df):
    c = df['Close']
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.squeeze()

def run():
    end   = datetime.today()
    start = end - timedelta(days=7500)  # ~20.5 years

    print('\n' + '='*60)
    print('  CACHE BUILDER — 20-YEAR PRICE HISTORY')
    print(f'  From: {start.date()}  To: {end.date()}')
    print('='*60 + '\n')

    data = {}
    failed = []

    for i, ticker in enumerate(UNIVERSE):
        try:
            df = flatten(yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=True
            ))
            if len(df) > 150:
                close = get_close(df)
                data[ticker] = close
                first = close.index[0].date()
                print(f'  [{i+1:2}/{len(UNIVERSE)}] {ticker:<20} {len(close):4} rows  from {first}')
            else:
                failed.append(ticker)
                print(f'  [{i+1:2}/{len(UNIVERSE)}] {ticker:<20} SKIP — only {len(df)} rows')
        except Exception as e:
            failed.append(ticker)
            print(f'  [{i+1:2}/{len(UNIVERSE)}] {ticker:<20} ERROR: {e}')

        time.sleep(random.uniform(0.4, 0.8))

    print(f'\n  Downloaded: {len(data)}/{len(UNIVERSE)} stocks')
    if failed:
        print(f'  Failed: {failed}')

    # Merge with existing Nifty 500 cache (adds the 450 recent-only stocks)
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'rb') as f:
                existing = pickle.load(f)
            added = 0
            for ticker, prices in existing.items():
                if ticker not in data:
                    data[ticker] = prices
                    added += 1
            print(f'  Merged {added} extra tickers from existing cache (recent period)')
        except Exception as e:
            print(f'  Could not merge existing cache: {e}')

    # Save
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(data, f)

    print(f'\n  Saved: {CACHE_PATH}')
    print(f'  Total stocks: {len(data)}')

    # Verify
    sample_ticker = 'HDFCBANK.NS'
    if sample_ticker in data:
        p = data[sample_ticker]
        print(f'\n  Verification: {sample_ticker}')
        print(f'  Start: {p.index[0].date()}')
        print(f'  End:   {p.index[-1].date()}')
        print(f'  Rows:  {len(p)}')
        if p.index[0].year > 2010:
            print(f'\n  WARNING: History only goes back to {p.index[0].year}.')
            print(f'  yfinance may have throttled the download.')
            print(f'  Wait 10 minutes and re-run to get full history.')
        else:
            print(f'\n  Full history confirmed. Ready to run 20-year backtest.')

    print('\n' + '='*60 + '\n')

if __name__ == '__main__':
    run()
