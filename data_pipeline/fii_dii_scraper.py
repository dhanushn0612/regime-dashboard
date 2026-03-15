"""
FII/DII Data Scraper — NSE India
=================================
Automatically downloads FII/DII activity data from NSE India
and feeds it into the Regime Classifier as a clean DataFrame.

Usage:
    python data_pipeline/fii_dii_scraper.py

Output:
    data_pipeline/fii_dii_data.csv  — cumulative historical data
    (auto-detected by run_classifier.py)

Schedule:
    Runs automatically as part of GitHub Actions daily pipeline.
    NSE updates FII/DII data after market close (~6:30 PM IST).
    Our pipeline runs at 10 AM IST next morning — always fresh.
"""

import os
import time
import random
import requests
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(SCRIPT_DIR, 'fii_dii_data.csv')

# ── NSE HEADERS (required — NSE blocks requests without these) ────────
HEADERS = {
    'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept':          'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer':         'https://www.nseindia.com/',
    'Connection':      'keep-alive',
    'sec-fetch-dest':  'empty',
    'sec-fetch-mode':  'cors',
    'sec-fetch-site':  'same-origin',
}

# ── NSE SESSION (must visit homepage first to get cookies) ────────────
def get_nse_session() -> requests.Session:
    """
    NSE requires a valid session cookie before API calls.
    Visits homepage first to obtain cookies.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    print("  Establishing NSE session...")
    try:
        # Visit homepage to get cookies
        session.get('https://www.nseindia.com', timeout=15)
        time.sleep(random.uniform(1.5, 2.5))  # Polite delay

        # Visit the FII/DII page to get page-specific cookies
        session.get('https://www.nseindia.com/reports/fii-dii', timeout=15)
        time.sleep(random.uniform(1.0, 2.0))

        print("  ✓ Session established")
        return session

    except Exception as e:
        print(f"  ✗ Session failed: {e}")
        return session


# ── FETCH SINGLE DATE ─────────────────────────────────────────────────
def fetch_fii_dii_for_date(session: requests.Session, date: datetime) -> dict | None:
    """
    Fetch FII/DII data for a single date from NSE API.
    Returns dict with date, FII_Net, DII_Net or None if unavailable.
    """
    date_str = date.strftime('%d-%m-%Y')

    url = f"https://www.nseindia.com/api/fiidiiTradeReact?date={date_str}"

    try:
        response = session.get(url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data or not isinstance(data, list):
            return None

        # NSE returns list of category objects
        # We want: "FII/FPI" and "DII" categories, "Net Purchase / Sales" column
        fii_net = None
        dii_net = None

        for item in data:
            category = item.get('category', '').strip().upper()
            net = item.get('netPurchaseSales', None)

            if net is None:
                # Try alternate field names
                net = item.get('net', item.get('netPurchase', None))

            if net is not None:
                # Clean the value — NSE sometimes sends strings with commas
                if isinstance(net, str):
                    net = float(net.replace(',', '').replace(' ', ''))
                else:
                    net = float(net)

                if 'FII' in category or 'FPI' in category:
                    fii_net = net
                elif 'DII' in category:
                    dii_net = net

        if fii_net is not None and dii_net is not None:
            return {
                'date':    date.strftime('%Y-%m-%d'),
                'FII_Net': fii_net,
                'DII_Net': dii_net,
            }

        return None

    except Exception:
        return None


# ── FETCH DATE RANGE ──────────────────────────────────────────────────
def fetch_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch FII/DII data for a date range.
    Skips weekends automatically.
    Adds polite delays to avoid rate limiting.
    """
    session = get_nse_session()
    records = []

    current = start_date
    total_days = (end_date - start_date).days
    fetched = 0
    skipped = 0

    print(f"  Fetching {total_days} calendar days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")

    while current <= end_date:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        result = fetch_fii_dii_for_date(session, current)

        if result:
            records.append(result)
            fetched += 1
        else:
            skipped += 1  # Holiday or no data

        # Polite delay — avoid hammering NSE
        time.sleep(random.uniform(0.3, 0.7))

        current += timedelta(days=1)

    print(f"  ✓ Fetched {fetched} trading days ({skipped} skipped — holidays/no data)")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ── INCREMENTAL UPDATE ────────────────────────────────────────────────
def update_fii_dii_data(lookback_days: int = 365) -> pd.DataFrame:
    """
    Smart incremental update:
    - If CSV exists: only fetch missing dates since last entry
    - If CSV doesn't exist: fetch full history (lookback_days)
    """
    end_date = datetime.today()

    if os.path.exists(OUTPUT_CSV):
        # Load existing data
        existing = pd.read_csv(OUTPUT_CSV, parse_dates=['date'])
        last_date = existing['date'].max()

        # Only fetch what's missing
        start_date = last_date + timedelta(days=1)
        print(f"  Existing data found up to {last_date.strftime('%Y-%m-%d')}")
        print(f"  Fetching incremental update from {start_date.strftime('%Y-%m-%d')}...")

        if start_date > end_date:
            print("  ✓ Already up to date")
            return existing

        new_data = fetch_range(start_date, end_date)

        if new_data.empty:
            print("  No new data available yet")
            return existing

        # Merge and deduplicate
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
        combined.sort_values('date', inplace=True)
        combined.reset_index(drop=True, inplace=True)

    else:
        # Full historical fetch
        start_date = end_date - timedelta(days=lookback_days)
        print(f"  No existing data. Fetching {lookback_days} days of history...")
        combined = fetch_range(start_date, end_date)

        if combined.empty:
            print("  ✗ Could not fetch any FII/DII data")
            return pd.DataFrame()

    # Save to CSV
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Saved {len(combined)} records to {OUTPUT_CSV}")

    return combined


# ── LOAD FOR CLASSIFIER ───────────────────────────────────────────────
def load_fii_dii() -> pd.DataFrame | None:
    """
    Load FII/DII data for use in the Regime Classifier.
    Returns DataFrame with columns: date, FII_Net, DII_Net
    Returns None if data unavailable.
    
    Call this from run_classifier.py:
        from fii_dii_scraper import load_fii_dii
        fii_dii_df = load_fii_dii()
        classifier.score(..., fii_dii_df=fii_dii_df)
    """
    if not os.path.exists(OUTPUT_CSV):
        return None

    df = pd.read_csv(OUTPUT_CSV, parse_dates=['date'])

    if df.empty:
        return None

    # Validate columns
    required = {'date', 'FII_Net', 'DII_Net'}
    if not required.issubset(df.columns):
        print(f"  ✗ CSV missing columns. Found: {list(df.columns)}")
        return None

    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ── DIAGNOSTICS ───────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    if df.empty:
        print("\n  No data to summarise")
        return

    print(f"\n{'='*55}")
    print(f"  FII/DII DATA SUMMARY")
    print(f"{'='*55}")
    print(f"  Records:       {len(df)} trading days")
    print(f"  Date Range:    {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  FII Net Range: {df['FII_Net'].min():,.0f} to {df['FII_Net'].max():,.0f} Cr")
    print(f"  DII Net Range: {df['DII_Net'].min():,.0f} to {df['DII_Net'].max():,.0f} Cr")

    # Recent 5 days
    print(f"\n  LAST 5 TRADING DAYS:")
    print(f"  {'Date':<14} {'FII Net (Cr)':>14} {'DII Net (Cr)':>14}")
    print(f"  {'-'*44}")
    for _, row in df.tail(5).iterrows():
        fii_str = f"{row['FII_Net']:>+,.0f}"
        dii_str = f"{row['DII_Net']:>+,.0f}"
        print(f"  {row['date'].strftime('%Y-%m-%d'):<14} {fii_str:>14} {dii_str:>14}")

    # 20-day rolling
    recent_fii = df['FII_Net'].tail(20).sum()
    recent_dii = df['DII_Net'].tail(20).sum()
    print(f"\n  20-DAY ROLLING:")
    print(f"  FII Net Flow: {recent_fii:>+,.0f} Cr  ({'BUYING' if recent_fii > 0 else 'SELLING'})")
    print(f"  DII Net Flow: {recent_dii:>+,.0f} Cr  ({'BUYING' if recent_dii > 0 else 'SELLING'})")
    print(f"{'='*55}\n")


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  FII/DII SCRAPER — {datetime.today().strftime('%Y-%m-%d %H:%M')} IST")
    print(f"{'='*55}\n")

    df = update_fii_dii_data(lookback_days=500)
    print_summary(df)
