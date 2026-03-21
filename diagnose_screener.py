import os, warnings, pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def s(x):
    if hasattr(x, 'item'): return float(x.item())
    return float(x)

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_close(df):
    c = df['Close']
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    return c.squeeze()

# ── Load stock data ───────────────────────────────────────────────────
price_cache = os.path.join(SCRIPT_DIR, "nifty500_prices.pkl")
if os.path.exists(price_cache):
    print(f"Loading Nifty 500 cache...")
    with open(price_cache, "rb") as f:
        stock_data = pickle.load(f)
    print(f"  {len(stock_data)} stocks loaded")
else:
    print("No cache found — downloading 10 stocks for diagnosis...")
    tickers = ["HDFCBANK.NS","TCS.NS","INFY.NS","HINDUNILVR.NS","RELIANCE.NS",
               "SUNPHARMA.NS","ITC.NS","MARUTI.NS","LT.NS","TITAN.NS"]
    stock_data = {}
    end = datetime.today()
    start = end - timedelta(days=2800)
    for t in tickers:
        try:
            df = flatten(yf.download(t, start=start, end=end, progress=False, auto_adjust=True))
            if len(df) > 100:
                stock_data[t] = get_close(df)
        except: pass
    print(f"  {len(stock_data)} stocks downloaded")

# ── Test dates — one Bull regime month from each year ─────────────────
TEST_DATES = [
    ("2020-08-31", 3, "MBull — should select ~8 stocks"),
    ("2021-06-30", 3, "MBull — should select ~10 stocks"),
    ("2022-10-31", 3, "MBull — should select ~10 stocks"),
    ("2023-06-30", 3, "MBull — should select ~10 stocks"),
    ("2024-06-30", 4, "SBull — should select ~15 stocks"),
]

print("\n" + "="*70)
print("  SCREENER DIAGNOSTIC")
print("="*70)

for date_str, regime_code, label in TEST_DATES:
    as_of = pd.Timestamp(date_str)
    print(f"\n--- {date_str} [{label}] ---")

    sma_floor = 0.90 if regime_code >= 3 else 0.95
    vol_cap   = {4: 60, 3: 70, 2: 75, 1: 65, 0: 55}.get(regime_code, 70)

    total       = 0
    fail_nodata = 0
    fail_history= 0
    fail_sma    = 0
    fail_vol    = 0
    passed      = 0

    vol_values  = []
    sma_ratios  = []

    for ticker, price in list(stock_data.items())[:50]:  # Check first 50
        total += 1

        # Slice to as_of date
        past = price[price.index <= as_of]

        if len(past) == 0:
            fail_nodata += 1
            continue

        if len(past) < 100:
            fail_history += 1
            continue

        try:
            curr  = s(past.iloc[-1])
            sma50 = s(past.rolling(50).mean().iloc[-1])
            ratio = curr / sma50
            sma_ratios.append(ratio)

            if ratio < sma_floor:
                fail_sma += 1
                continue

            rets = past.pct_change().dropna()
            if len(rets) < 20:
                fail_history += 1
                continue

            vol = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
            vol_values.append(vol)

            if vol > vol_cap:
                fail_vol += 1
                continue

            passed += 1

        except Exception as e:
            fail_nodata += 1

    print(f"  Stocks checked:      {total}")
    print(f"  No data at this date:{fail_nodata}")
    print(f"  Too short history:   {fail_history}")
    print(f"  Failed SMA filter:   {fail_sma}  (need price >= {sma_floor*100:.0f}% of SMA50)")
    print(f"  Failed vol filter:   {fail_vol}  (need vol <= {vol_cap}%)")
    print(f"  PASSED all filters:  {passed}")

    if sma_ratios:
        arr = np.array(sma_ratios)
        print(f"\n  SMA ratio stats (price/SMA50):")
        print(f"    Min: {arr.min():.3f}  Max: {arr.max():.3f}  Median: {np.median(arr):.3f}")
        pct_above_floor = (arr >= sma_floor).mean() * 100
        print(f"    % above {sma_floor} floor: {pct_above_floor:.0f}%")

    if vol_values:
        arr = np.array(vol_values)
        print(f"\n  Volatility stats (annualised %):")
        print(f"    Min: {arr.min():.1f}%  Max: {arr.max():.1f}%  Median: {np.median(arr):.1f}%")
        pct_below_cap = (arr <= vol_cap).mean() * 100
        print(f"    % below {vol_cap}% cap: {pct_below_cap:.0f}%")

# ── Deep dive on one specific stock ──────────────────────────────────
print("\n" + "="*70)
print("  DEEP DIVE: HDFCBANK.NS at 2020-08-31")
print("="*70)

if "HDFCBANK.NS" in stock_data:
    as_of = pd.Timestamp("2020-08-31")
    price = stock_data["HDFCBANK.NS"]
    past  = price[price.index <= as_of]

    print(f"  Total rows in cache:     {len(price)}")
    print(f"  Rows up to 2020-08-31:   {len(past)}")
    if len(past) > 0:
        print(f"  First date in slice:     {past.index[0].date()}")
        print(f"  Last date in slice:      {past.index[-1].date()}")
        print(f"  Price on 2020-08-31:     {s(past.iloc[-1]):.2f}")

        if len(past) >= 50:
            sma50 = s(past.rolling(50).mean().iloc[-1])
            curr  = s(past.iloc[-1])
            ratio = curr / sma50
            print(f"  SMA50 on 2020-08-31:     {sma50:.2f}")
            print(f"  Price/SMA50 ratio:       {ratio:.3f}")
            print(f"  SMA filter (need ≥0.90): {'PASS' if ratio >= 0.90 else 'FAIL'}")

        rets = past.pct_change().dropna()
        if len(rets) >= 20:
            vol = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
            print(f"  20-day Ann. Vol:         {vol:.1f}%")
            print(f"  Vol filter (need ≤70%):  {'PASS' if vol <= 70 else 'FAIL'}")
    else:
        print("  *** NO DATA available up to 2020-08-31 ***")
        print(f"  Cache starts at: {price.index[0].date()}")
        print(f"  This means the cache only has recent data — the pickle was")
        print(f"  built recently and yfinance only returned ~2 years of history.")
else:
    print("  HDFCBANK.NS not in stock_data")

print("\n" + "="*70)
print("  CACHE DATE RANGE CHECK (first 10 stocks)")
print("="*70)
for ticker, price in list(stock_data.items())[:10]:
    as_of_2020 = pd.Timestamp("2020-08-31")
    past_2020  = price[price.index <= as_of_2020]
    print(f"  {ticker:<20} Cache: {price.index[0].date()} to {price.index[-1].date()} | "
          f"Rows at 2020-08-31: {len(past_2020)}")

print("\nDiagnosis complete.")
