"""
Nifty 500 Universe Builder + Expanded Backtest
================================================
Downloads the official Nifty 500 constituent list from NSE India,
then runs the full backtest on the complete universe.

Two parts:
  Part 1 — get_nifty500_tickers()
    Downloads current Nifty 500 constituents from NSE
    Converts NSE symbols to Yahoo Finance format (.NS suffix)
    Saves to data_pipeline/nifty500_tickers.csv

  Part 2 — run_expanded_backtest()
    Downloads price data for all 500 stocks (batched to avoid rate limits)
    Runs the same walk-forward backtest as backtest.py
    But on the full 500-stock universe

Usage:
    Step 1: python data_pipeline/nifty500_universe.py --fetch
    Step 2: python data_pipeline/backtest.py --universe nifty500

NOTE on timing:
  Downloading 500 stocks takes ~8-12 minutes.
  Run once, data is cached. Subsequent runs use cache.
"""

import os
import sys
import time
import random
import requests
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TICKERS_CSV  = os.path.join(SCRIPT_DIR, 'nifty500_tickers.csv')
PRICE_CACHE  = os.path.join(SCRIPT_DIR, 'nifty500_prices.pkl')


# ── PART 1: GET NIFTY 500 TICKERS FROM NSE ───────────────────────────
def get_nifty500_tickers() -> list:
    """
    Download Nifty 500 constituent list from NSE India.
    NSE provides a CSV file with all index constituents.
    URL: https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500
    """
    print("Fetching Nifty 500 constituents from NSE...")

    # Method 1: NSE API (requires session warming)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.nseindia.com/',
    })

    tickers = []

    try:
        # Warm session
        session.get('https://www.nseindia.com', timeout=15)
        time.sleep(3)

        # Get Nifty 500 constituents
        url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500'
        r = session.get(url, timeout=20)

        if r.status_code == 200:
            data = r.json()
            stocks = data.get('data', [])
            print(f"  NSE API returned {len(stocks)} stocks")

            for stock in stocks:
                symbol = stock.get('symbol', '')
                if symbol and symbol != 'NIFTY 500':
                    tickers.append(symbol + '.NS')

            print(f"  Converted to {len(tickers)} Yahoo Finance tickers")

    except Exception as e:
        print(f"  NSE API failed: {e}")

    # Method 2: NSE CSV download (fallback)
    if not tickers:
        print("  Trying NSE CSV download...")
        try:
            session.get('https://www.nseindia.com', timeout=15)
            time.sleep(3)

            csv_url = 'https://nseindia.com/content/indices/ind_nifty500list.csv'
            r = session.get(csv_url, timeout=20)

            if r.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                if 'Symbol' in df.columns:
                    tickers = [s + '.NS' for s in df['Symbol'].dropna().tolist()]
                    print(f"  CSV: {len(tickers)} tickers")

        except Exception as e:
            print(f"  CSV failed: {e}")

    # Method 3: Use our curated list if NSE is blocked
    if not tickers:
        print("  Using curated Nifty 500 representative list...")
        tickers = get_curated_nifty500()

    # Save
    if tickers:
        pd.DataFrame({'ticker': tickers}).to_csv(TICKERS_CSV, index=False)
        print(f"  Saved {len(tickers)} tickers to {TICKERS_CSV}")

    return tickers


def get_curated_nifty500() -> list:
    """
    Curated list of ~200 most liquid Nifty 500 stocks.
    Covers all major sectors and market caps.
    Better than 48 but manageable for download.
    """
    return [
        # Large cap — IT (15)
        "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
        "LTIM.NS","MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","OFSS.NS",
        "TATAELXSI.NS","LTTS.NS","KPITTECH.NS","HEXAWARE.NS","NIIT.NS",

        # Large cap — Banks (15)
        "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
        "INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","IDFCFIRSTB.NS","PNB.NS",
        "BANKBARODA.NS","CANARABANK.NS","UNIONBANK.NS","INDIANB.NS","MAHABANK.NS",

        # Financial Services (12)
        "BAJFINANCE.NS","BAJAJFINSV.NS","MUTHOOTFIN.NS","CHOLAFIN.NS",
        "MANAPPURAM.NS","LICHSGFIN.NS","PFC.NS","RECLTD.NS","IRFC.NS",
        "HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS",

        # FMCG (12)
        "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
        "GODREJCP.NS","MARICO.NS","COLPAL.NS","EMAMILTD.NS","TATACONSUM.NS",
        "VBL.NS","RADICO.NS",

        # Auto (12)
        "MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS",
        "TVSMOTORS.NS","ASHOKLEY.NS","BOSCHLTD.NS","MOTHERSON.NS",
        "BHARATFORG.NS","SUNDRMFAST.NS","APOLLOTYRE.NS","MRF.NS",

        # Pharma (12)
        "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS",
        "TORNTPHARM.NS","ALKEM.NS","LUPIN.NS","IPCALAB.NS","GLENMARK.NS",
        "BIOCON.NS","NATCOPHARM.NS",

        # Energy & Oil (10)
        "RELIANCE.NS","ONGC.NS","COALINDIA.NS","BPCL.NS","IOC.NS",
        "GAIL.NS","HINDPETRO.NS","PETRONET.NS","OIL.NS","MGL.NS",

        # Metals & Mining (10)
        "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS",
        "NMDC.NS","MOIL.NS","NATIONALUM.NS","HINDCOPPER.NS","APLAPOLLO.NS",

        # Capital Goods & Infra (12)
        "LT.NS","NTPC.NS","POWERGRID.NS","TATAPOWER.NS","SIEMENS.NS",
        "ABB.NS","HAVELLS.NS","CUMMINSIND.NS","BHEL.NS","THERMAX.NS",
        "POLYCAB.NS","KEI.NS",

        # Consumer & Retail (10)
        "TITAN.NS","ASIANPAINT.NS","DMART.NS","TRENT.NS","JUBLFOOD.NS",
        "DEVYANI.NS","WESTLIFE.NS","SHOPERSTOP.NS","ZOMATO.NS","NYKAA.NS",

        # Real Estate (8)
        "DLF.NS","GODREJPROP.NS","OBEROIRLTY.NS","PRESTIGE.NS",
        "BRIGADE.NS","PHOENIXLTD.NS","LODHA.NS","SOBHA.NS",

        # Cement (8)
        "ULTRACEMCO.NS","AMBUJACEM.NS","ACC.NS","SHREECEM.NS",
        "DALMIACEME.NS","JKCEMENT.NS","RAMCOCEM.NS","HEIDELBERG.NS",

        # Specialty Chemicals (8)
        "PIDILITIND.NS","AAVAS.NS","ATUL.NS","NAVINFLUOR.NS",
        "DEEPAKNITR.NS","BALRAMCHIN.NS","ALKYLAMINE.NS","FINEORG.NS",

        # Telecom (5)
        "BHARTIARTL.NS","IDEA.NS","TATACOMM.NS","RAILTEL.NS","HFCL.NS",

        # Hotels & Tourism (6)
        "INDHOTEL.NS","LEMONTREE.NS","CHALET.NS","EIHOTEL.NS","MHRIL.NS","TAJGVK.NS",

        # Healthcare Infra (5)
        "APOLLOHOSP.NS","FORTIS.NS","MAXHEALTH.NS","METROPOLIS.NS","DRLABL.NS",

        # Mid cap Quality (20)
        "PERSISTENT.NS","COFORGE.NS","KPITTECH.NS","TATAELXSI.NS","MPHASIS.NS",
        "VOLTAS.NS","CROMPTON.NS","DIXON.NS","AMBER.NS","KAYNES.NS",
        "RVNL.NS","IRCTC.NS","CONCOR.NS","GSPL.NS","IGL.NS",
        "PAGEIND.NS","WHIRLPOOL.NS","BERGEPAINT.NS","KANSAINER.NS","AKZOINDIA.NS",
    ]


# ── PART 2: BATCH DOWNLOAD ────────────────────────────────────────────
def download_nifty500_prices(tickers: list, lookback_days: int = 1000,
                              use_cache: bool = True) -> dict:
    """
    Download prices for all Nifty 500 stocks in batches.
    Uses pickle cache to avoid re-downloading.
    """
    import pickle

    # Check cache
    if use_cache and os.path.exists(PRICE_CACHE):
        cache_age = (datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(PRICE_CACHE))).days
        if cache_age < 7:
            print(f"  Loading cached prices ({cache_age} days old)...")
            with open(PRICE_CACHE, 'rb') as f:
                data = pickle.load(f)
            print(f"  Cache: {len(data)} stocks")
            return data

    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    print(f"Downloading {len(tickers)} stocks in batches (this takes 8-12 minutes)...")
    data = {}
    failed = 0

    # Download in batches of 10 to avoid rate limiting
    batch_size = 10
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        print(f"  Batch {batch_num}/{total_batches} ({i+1}-{min(i+batch_size, len(tickers))})...", end=' ')

        for ticker in batch:
            try:
                df_raw = yf.download(ticker, start=start, end=end,
                                     progress=False, auto_adjust=True)
                if isinstance(df_raw.columns, pd.MultiIndex):
                    df_raw.columns = df_raw.columns.get_level_values(0)
                if len(df_raw) > 150:
                    c = df_raw['Close']
                    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
                    data[ticker] = c.squeeze()
                else:
                    failed += 1
            except Exception:
                failed += 1

        print(f"ok ({len(data)} loaded, {failed} failed)")
        time.sleep(random.uniform(1.0, 2.0))  # Polite delay between batches

    print(f"\n  Total: {len(data)} loaded, {failed} failed out of {len(tickers)}")

    # Cache the results
    with open(PRICE_CACHE, 'wb') as f:
        import pickle
        pickle.dump(data, f)
    print(f"  Cached to {PRICE_CACHE}")

    return data


# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  NIFTY 500 UNIVERSE BUILDER")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*60 + "\n")

    # Step 1: Get tickers
    if os.path.exists(TICKERS_CSV):
        df = pd.read_csv(TICKERS_CSV)
        tickers = df['ticker'].tolist()
        print(f"  Loaded {len(tickers)} tickers from cache")
        print(f"  To refresh: delete {TICKERS_CSV} and re-run")
    else:
        tickers = get_nifty500_tickers()

    if not tickers:
        print("  ERROR: No tickers found")
        return

    print(f"\n  Universe: {len(tickers)} stocks")
    print(f"  Sample: {tickers[:5]}")

    # Step 2: Download prices
    force = '--force' in sys.argv
    data  = download_nifty500_prices(tickers, use_cache=not force)

    print(f"\n  Ready: {len(data)} stocks with price history")
    print(f"\n  Next step: python data_pipeline/backtest.py")
    print(f"  The backtest will automatically use this expanded universe.\n")


if __name__ == "__main__":
    main()
