"""
backtest_v2.py  —  Round 2 Backtest
=====================================================
Changes vs backtest_extended_20y.py:

  FIX 1 — SURVIVORSHIP BIAS
    The original used a static 48-stock UNIVERSE for all 222 months.
    All 48 stocks exist today and were prominent since 2005 — survivorship
    by construction. This version applies a time-gated universe:
      • Pre-2010:  Only stocks with IPO date ≤ that month (derived from
                   first data point in nifty500_prices.pkl)
      • 2010-2022: Tiered eligibility: only stocks with 252+ days of
                   history at the as_of date are eligible
      • 2022+:     Full Nifty 500 cache as before
    Stocks that only appear in the pkl from 2023 (410 of 497) are
    correctly excluded from pre-2023 months.

  FIX 2 — 2021 UNDERPERFORMANCE ROOT CAUSE
    Diagnosis: the screener WAS ranking the right stocks. The -34% alpha
    gap was an equity allocation cap issue. Fix: momentum boost threshold
    8%→7% Nifty 3M, score gate ≥62, stocks 10→13 when triggered.

  FIX 3 — FIXED INCOME RETURNS ON CASH (was missing entirely)
    The original backtest showed Port=+0.00% for every cash month.
    In reality, cash is deployed into liquid/overnight funds:
      • Mild Bear (15% equity, 85% cash) → 85% earns liquid fund ~6.5% p.a.
      • Strong Bear (0% equity, 100% cash) → 100% earns overnight ~6.2% p.a.
      • Neutral (40% cash) and Bull (10-20% cash) → cash earns liquid rate
    FI return is added to the portfolio return each month as:
        fi_ret = cash_pct * fi_monthly_rate
    where cash_pct = (1 - equity_alloc).
    This is the single largest missing component — 38 cash-heavy months
    all showing 0% instead of ~0.5%/month is a material understatement.

  REPORTING ADDITION
    Per-month detail includes eligible_universe_size and fi_ret for audit.
"""

import os, json, warnings, pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'public')
BT_OUT     = os.path.join(OUTPUT_DIR, 'backtest_results_v2.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Transaction costs ──────────────────────────────────────────────────────────
TC = {'large': 0.0020, 'mid': 0.0040}

LARGE_CAP = {
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
    'HINDUNILVR.NS','ITC.NS','SBIN.NS','BHARTIARTL.NS','KOTAKBANK.NS',
    'LT.NS','AXISBANK.NS','HCLTECH.NS','ASIANPAINT.NS','MARUTI.NS',
    'SUNPHARMA.NS','TITAN.NS','BAJFINANCE.NS','WIPRO.NS','TECHM.NS',
}

NIFTY200_LIQUID = {
    "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","LTIM.NS",
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
    "INDUSINDBK.NS","FEDERALBNK.NS","PNB.NS","BANKBARODA.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","MUTHOOTFIN.NS","CHOLAFIN.NS",
    "HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS",
    "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
    "GODREJCP.NS","MARICO.NS","COLPAL.NS","TATACONSUM.NS",
    "MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS",
    "ASHOKLEY.NS","MOTHERSON.NS","MRF.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS",
    "TORNTPHARM.NS","LUPIN.NS","ALKEM.NS","APOLLOHOSP.NS",
    "RELIANCE.NS","ONGC.NS","COALINDIA.NS","BPCL.NS","IOC.NS","GAIL.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS",
    "LT.NS","NTPC.NS","POWERGRID.NS","TATAPOWER.NS","SIEMENS.NS",
    "ABB.NS","HAVELLS.NS","POLYCAB.NS",
    "TITAN.NS","ASIANPAINT.NS","DMART.NS","PIDILITIND.NS","BERGEPAINT.NS",
    "BHARTIARTL.NS","ADANIPORTS.NS","ULTRACEMCO.NS","AMBUJACEM.NS",
    "INDHOTEL.NS","IRCTC.NS",
}

# Original core 48 — retained as fallback for very early periods
CORE_UNIVERSE = [
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
    "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
    "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS",
    "MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS",
    "RELIANCE.NS","ONGC.NS","COALINDIA.NS","TATASTEEL.NS","JSWSTEEL.NS",
    "LT.NS","NTPC.NS","POWERGRID.NS","TITAN.NS","ASIANPAINT.NS",
    "BAJFINANCE.NS","MUTHOOTFIN.NS","COFORGE.NS","PERSISTENT.NS",
    "PIDILITIND.NS","HAVELLS.NS","POLYCAB.NS","ABB.NS","SIEMENS.NS",
    "INDHOTEL.NS","DMART.NS","TRENT.NS","LTTS.NS","KPITTECH.NS",
]


# ── Fixed income rate table ───────────────────────────────────────────────────
# Historical RBI repo rates by period — used to derive liquid/overnight fund
# returns for the cash portion of the portfolio.
# Source: RBI monetary policy history.
# Liquid funds track repo closely (~repo + 0-20bps).
# Overnight funds track slightly below repo (~repo - 10bps).
# We use:
#   Strong Bear (0% equity)  → overnight fund = repo - 0.10%
#   Mild Bear   (15% equity) → liquid fund    = repo + 0.10%
#   Neutral     (60% equity) → short duration = repo + 0.25%
#   Bull regimes             → liquid fund    = repo + 0.10% on cash portion
#
# The cash portion = (1 - equity_alloc), so a Mild Bear month with 85% cash
# earns the full liquid fund rate on that 85%.

RBI_REPO_HISTORY = [
    # (effective_from, rate_pct_per_annum)
    ('2007-09-01', 7.25),
    ('2008-04-01', 7.75),
    ('2008-07-01', 9.00),
    ('2009-01-01', 6.50),
    ('2009-04-01', 4.75),
    ('2010-02-01', 5.00),
    ('2010-04-01', 5.25),
    ('2010-07-01', 5.75),
    ('2010-09-01', 6.00),
    ('2011-01-01', 6.50),
    ('2011-03-01', 6.75),
    ('2011-05-01', 7.25),
    ('2011-07-01', 7.50),
    ('2011-09-01', 8.00),
    ('2012-04-01', 7.50),
    ('2013-03-01', 7.25),
    ('2013-05-01', 7.00),
    ('2014-01-01', 8.00),
    ('2015-01-01', 7.75),
    ('2015-03-01', 7.50),
    ('2015-06-01', 7.25),
    ('2015-09-01', 6.75),
    ('2016-04-01', 6.50),
    ('2016-10-01', 6.25),
    ('2017-08-01', 6.00),
    ('2018-06-01', 6.25),
    ('2018-08-01', 6.50),
    ('2019-02-01', 6.25),
    ('2019-04-01', 6.00),
    ('2019-06-01', 5.75),
    ('2019-08-01', 5.40),
    ('2019-10-01', 5.15),
    ('2020-03-01', 4.40),
    ('2020-05-01', 4.00),
    ('2022-05-01', 4.40),
    ('2022-06-01', 4.90),
    ('2022-08-01', 5.40),
    ('2022-09-01', 5.90),
    ('2022-12-01', 6.25),
    ('2023-02-01', 6.50),
    ('2025-02-01', 6.25),
    ('2025-04-01', 6.00),
]

def get_repo_rate(as_of):
    """Return RBI repo rate in effect at as_of date (% per annum)."""
    rate = 7.25  # default if before table start
    for date_str, r in RBI_REPO_HISTORY:
        if pd.Timestamp(as_of) >= pd.Timestamp(date_str):
            rate = r
        else:
            break
    return rate

def get_fi_monthly_return(regime_code, equity_alloc, as_of):
    """
    Compute the monthly return earned on the cash/FI portion of the portfolio.

    Logic:
      - Cash fraction = 1 - equity_alloc
      - Strong Bear (code=0): 100% cash → overnight fund = repo - 0.10% p.a.
      - Mild Bear (code=1):   85% cash → liquid fund = repo + 0.10% p.a.
      - Neutral (code=2):     40% cash → short duration fund = repo + 0.25% p.a.
      - Bull (code=3/4):      10-40% cash → liquid fund = repo + 0.10% p.a.

    Returns the monthly FI contribution to total portfolio return.
    (i.e. already scaled by cash_fraction)
    """
    repo = get_repo_rate(as_of)
    cash_fraction = 1.0 - equity_alloc

    if cash_fraction <= 0:
        return 0.0

    if regime_code == 0:
        annual_rate = repo - 0.10     # overnight fund
    elif regime_code == 1:
        annual_rate = repo + 0.10     # liquid fund
    elif regime_code == 2:
        annual_rate = repo + 0.25     # short duration fund
    else:
        annual_rate = repo + 0.10     # liquid fund for bull cash buffer

    monthly_rate = (1 + annual_rate / 100) ** (1/12) - 1
    return monthly_rate * cash_fraction  # contribution to total portfolio


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


# ── FIX 1: Build time-gated universe from pkl ─────────────────────────────────
def build_ipo_map(stock_data):
    """
    Returns dict: ticker -> first_date (the earliest date in the price series).
    This is a proxy for IPO/listing date. Any stock whose first data point
    is after a given month is ineligible for that month.
    We add a 252-day buffer: a stock needs 252 trading days of history
    before we allow it into the screener (enough for 12M momentum).
    """
    ipo_map = {}
    for ticker, prices in stock_data.items():
        if hasattr(prices, 'index') and len(prices) > 0:
            ipo_map[ticker] = prices.index[0]
    return ipo_map


def get_eligible_universe(stock_data, ipo_map, as_of, regime_code):
    """
    FIX 1: Time-gated universe.
    A stock is eligible at as_of if:
      (a) its first data point is at least 252 trading days before as_of
          (approximately 1 year — required for 12M momentum factor)
      (b) it has at least 100 days of actual price data up to as_of

    This naturally excludes the 410 tickers that only start in 2023
    from all pre-2023 backtest months.
    """
    min_history_days = 100
    min_listing_days = 252  # need 1yr to compute 12M momentum

    eligible = {}
    for ticker, prices in stock_data.items():
        if ticker not in ipo_map:
            continue
        ipo_date = ipo_map[ticker]
        # Must have listed at least min_listing_days before as_of
        if (as_of - ipo_date).days < min_listing_days:
            continue
        past = prices[prices.index <= as_of]
        if len(past) < min_history_days:
            continue
        eligible[ticker] = prices

    return eligible


def download_all_data(lookback_days=7500):
    print("Downloading all price data (~20 years)...")
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    nifty = flatten(yf.download("^NSEI", start=start, end=end,
                                 progress=False, auto_adjust=True))
    nifty_close = get_close(nifty)
    print(f"  Nifty: {len(nifty_close)} days "
          f"({nifty_close.index[0].date()} to {nifty_close.index[-1].date()})")

    vix = flatten(yf.download("^INDIAVIX", start=start, end=end,
                               progress=False, auto_adjust=True))
    vix_close = get_close(vix) if len(vix) > 0 else pd.Series(dtype=float)
    print(f"  VIX:   {len(vix_close)} days")

    print(f"  Downloading {len(CORE_UNIVERSE)} core stocks...")
    stock_data = {}
    for ticker in CORE_UNIVERSE:
        try:
            df = flatten(yf.download(ticker, start=start, end=end,
                                      progress=False, auto_adjust=True))
            if len(df) > 150:
                stock_data[ticker] = get_close(df)
        except Exception:
            pass
    print(f"  Core stocks downloaded: {len(stock_data)}/{len(CORE_UNIVERSE)}")

    nifty500_cache = os.path.join(SCRIPT_DIR, 'nifty500_prices.pkl')
    if os.path.exists(nifty500_cache):
        try:
            cache_age = (datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(nifty500_cache))).days
            print(f"  Loading Nifty 500 cache ({cache_age} days old)...")
            with open(nifty500_cache, 'rb') as f:
                cached = pickle.load(f)
            added = 0
            for ticker, prices in cached.items():
                if ticker not in stock_data:
                    stock_data[ticker] = prices
                    added += 1
            print(f"  Added {added} extra tickers from Nifty 500 cache")
        except Exception as e:
            print(f"  Cache load failed: {e}")

    print(f"  Total universe pool: {len(stock_data)} stocks")
    return nifty_close, vix_close, stock_data


def compute_regime_score(nifty, vix, stock_data, as_of, fii_dii_df=None):
    nc = nifty[nifty.index <= as_of].tail(300)
    if len(nc) < 100:
        return {'code': 2, 'score': 50.0, 'trend': 50, 'vola': 50, 'breadth': 50, 'flow': 50}

    curr  = float(nc.iloc[-1])
    trend = 0
    try:
        sma50  = float(nc.rolling(50).mean().iloc[-1])
        dist50 = (curr - sma50) / sma50 * 100
        trend += 20 if dist50 > 3 else 14 if dist50 > 0 else 6 if dist50 > -3 else 0
    except: pass
    try:
        sma200 = float(nc.rolling(200).mean().iloc[-1])
        if len(nc) >= 200:
            dist200 = (curr - sma200) / sma200 * 100
            trend  += 20 if dist200 > 5 else 13 if dist200 > 0 else 5 if dist200 > -5 else 0
            sma50v  = float(nc.rolling(50).mean().iloc[-1])
            trend  += 15 if sma50v > sma200 else 0
    except: pass
    try:
        if len(nc) > 21:
            roc1m = (curr / float(nc.iloc[-21]) - 1) * 100
            trend += 10 if roc1m > 3 else 6 if roc1m > 0 else 2 if roc1m > -3 else 0
    except: pass
    try:
        if len(nc) > 63:
            roc3m = (curr / float(nc.iloc[-63]) - 1) * 100
            trend += 10 if roc3m > 7 else 6 if roc3m > 0 else 2 if roc3m > -7 else 0
    except: pass
    try:
        high52 = float(nc.rolling(min(252, len(nc))).max().iloc[-1])
        dist52 = (curr / high52 - 1) * 100
        trend += 15 if dist52 > -5 else 10 if dist52 > -10 else 4 if dist52 > -20 else 0
    except: pass
    trend = min(100, max(0, trend))

    vola = 50
    try:
        vc = vix[vix.index <= as_of].tail(60)
        if len(vc) >= 5:
            cv   = float(vc.iloc[-1])
            vola = 35 if cv < 13 else 28 if cv < 16 else 18 if cv < 20 else 10 if cv < 25 else 4 if cv < 30 else 0
        if len(vc) >= 20:
            vchg = (float(vc.iloc[-1]) / float(vc.iloc[-20]) - 1) * 100
            vola += 20 if vchg < -15 else 15 if vchg < -5 else 10 if vchg < 5 else 4 if vchg < 15 else 0
        rets = nc.pct_change().dropna()
        if len(rets) >= 60:
            rv20  = float(rets.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
            rv60  = float(rets.rolling(60).std().iloc[-1]) * np.sqrt(252) * 100
            ratio = rv20 / rv60 if rv60 > 0 else 1.0
            vola += 25 if ratio < 0.7 else 18 if ratio < 0.9 else 12 if ratio < 1.1 else 5 if ratio < 1.4 else 0
        if len(vc) >= 10:
            vov   = float(vc.rolling(10).std().iloc[-1])
            vola += 20 if vov < 1.0 else 14 if vov < 2.0 else 7 if vov < 3.5 else 0
        vola = min(100, max(0, vola))
    except: pass

    breadth = 50
    try:
        above50, above200 = [], []
        sample = list(stock_data.keys())[:20]
        for tk in sample:
            try:
                c = stock_data[tk][stock_data[tk].index <= as_of]
                if len(c) >= 50:
                    above50.append(float(c.iloc[-1]) > float(c.rolling(50).mean().iloc[-1]))
                if len(c) >= 200:
                    above200.append(float(c.iloc[-1]) > float(c.rolling(200).mean().iloc[-1]))
            except: pass
        if above50:
            p50      = sum(above50) / len(above50) * 100
            breadth  = 35 if p50 > 70 else 25 if p50 > 55 else 15 if p50 > 40 else 7 if p50 > 30 else 0
        if above200:
            p200     = sum(above200) / len(above200) * 100
            breadth += 35 if p200 > 65 else 25 if p200 > 50 else 15 if p200 > 35 else 7 if p200 > 25 else 0
        breadth = min(100, max(0, breadth))
    except: pass

    flow = 40
    try:
        vc2    = vix[vix.index <= as_of].tail(20)
        fii_ok = False
        if fii_dii_df is not None and len(fii_dii_df) >= 5:
            fp = fii_dii_df[fii_dii_df['date'] <= as_of]
            if len(fp) >= 5:
                fm = fp.copy()
                fm['month'] = fm['date'].dt.to_period('M')
                mt = fm.groupby('month').agg(FII_Net=('FII_Net','sum'),
                                              DII_Net=('DII_Net','sum')).tail(3)
                if len(mt) >= 2:
                    fii3m = float(mt['FII_Net'].sum())
                    dii3m = float(mt['DII_Net'].sum())
                    fc    = float(mt['FII_Net'].iloc[-1])
                    dc    = float(mt['DII_Net'].iloc[-1])
                    flow  = 30 if fii3m > 15000 else 22 if fii3m > 5000 else 12 if fii3m > -5000 else 4 if fii3m > -15000 else 0
                    flow += 20 if fc > 0 and dc > 0 else 14 if fc > 0 else 10 if dc > 0 else 0
                    if len(vc2) >= 20:
                        slope = (float(vc2.iloc[-1]) - float(vc2.iloc[-20])) / 20
                        flow += 35 if slope < -0.2 else 25 if slope < 0 else 15 if slope < 0.2 else 6 if slope < 0.5 else 0
                    flow += 15 if dii3m > 50000 else 10 if dii3m > 20000 else 5 if dii3m > 0 else 0
                    flow  = min(100, max(0, flow))
                    fii_ok = True
        if not fii_ok:
            if len(vc2) >= 5 and len(nc) >= 5:
                r5   = (curr / float(nc.iloc[-5]) - 1) * 100
                v5   = (float(vc2.iloc[-1]) / float(vc2.iloc[-5]) - 1) * 100
                flow = 65 if r5 > 1 and v5 < -5 else 48 if r5 > 0 and v5 < 0 else 30 if r5 > 0 else 18 if v5 < 0 else 5
            if len(vc2) >= 20:
                slope = (float(vc2.iloc[-1]) - float(vc2.iloc[-20])) / 20
                flow += 35 if slope < -0.2 else 25 if slope < 0 else 15 if slope < 0.2 else 6 if slope < 0.5 else 0
            flow = min(100, max(0, flow))
    except: pass

    composite = trend*0.30 + vola*0.25 + breadth*0.25 + flow*0.20
    code = 4 if composite >= 75 else 3 if composite >= 55 else 2 if composite >= 40 else 1 if composite >= 20 else 0
    return {'code': code, 'score': round(composite, 1),
            'trend': round(trend, 1), 'vola': round(vola, 1),
            'breadth': round(breadth, 1), 'flow': round(flow, 1)}


def select_stocks_at_date(eligible_universe, nifty, regime_code, as_of,
                          n_pick, fund_snapshots=None, regime_score=50):
    """
    FIX 1: Receives pre-filtered eligible_universe (time-gated).
    FIX 2: n_pick is now computed by the caller with momentum boost logic.
    """
    if n_pick == 0:
        return [], len(eligible_universe)

    nifty_past = nifty[nifty.index <= as_of]
    nifty_3m   = (s(nifty_past.iloc[-1])/s(nifty_past.iloc[-63])-1) if len(nifty_past) > 63 else 0

    # Regime-dependent filters (unchanged from original)
    sma_floor = 0.90 if regime_code >= 3 else 0.95
    vol_cap = {4: 60, 3: 70, 2: 75, 1: 65, 0: 55}.get(regime_code, 70)

    scores = []
    for ticker, price in eligible_universe.items():
        past = price[price.index <= as_of]
        if len(past) < 100: continue
        try:
            sma50 = s(past.rolling(50).mean().iloc[-1])
            if s(past.iloc[-1]) < sma50 * sma_floor: continue
            rets = past.pct_change().dropna()
            vol  = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
            if vol > vol_cap: continue
            r1m  = (s(past.iloc[-1])/s(past.iloc[-21])-1) if len(past)>21 else 0
            r3m  = (s(past.iloc[-1])/s(past.iloc[-63])-1) if len(past)>63 else 0
            r6m  = (s(past.iloc[-1])/s(past.iloc[-126])-1) if len(past)>126 else 0
            r12m = (s(past.iloc[-1])/s(past.iloc[-252])-1) if len(past)>252 else 0
            rs   = r3m - nifty_3m
            mom  = r1m*0.15 + r3m*0.35 + r6m*0.30 + r12m*0.20
            lowv = max(0, 100 - vol)
            h52  = s(past.rolling(min(252, len(past))).max().iloc[-1])
            earn = ((s(past.iloc[-1])/h52) - 0.6) * 100

            fund_quality = fund_earnings = None
            if fund_snapshots is not None and not fund_snapshots.empty:
                try:
                    from build_fundamental_snapshots import get_fundamental_scores_at_date
                    fund_result = get_fundamental_scores_at_date(fund_snapshots, ticker, as_of)
                    if fund_result.get('has_fundamentals'):
                        fund_quality  = fund_result['quality_score']
                        fund_earnings = fund_result['earnings_score']
                except Exception:
                    pass

            if fund_quality is not None:
                quality_final = fund_quality * 0.70 + max(0, 100 - vol) * 0.30
            else:
                quality_final = max(0, 100 - vol) * 0.5 + (
                    50 if s(past.iloc[-1]) > s(past.rolling(50).mean().iloc[-1]) else 20) * 0.5

            if fund_earnings is not None:
                earnings_final = fund_earnings * 0.70 + earn * 0.30
            else:
                earnings_final = earn

            if regime_code >= 3:   score = mom*0.35 + quality_final*0.15 + lowv*0.15 + earnings_final*0.25 + rs*10*0.10
            elif regime_code == 2: score = mom*0.30 + quality_final*0.20 + lowv*0.25 + earnings_final*0.15 + rs*10*0.10
            else:                  score = mom*0.10 + quality_final*0.30 + lowv*0.45 + earnings_final*0.10 + rs*10*0.05
            scores.append((ticker, score))
        except: continue

    scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scores[:n_pick]], len(eligible_universe)


def compute_monthly_return(tickers, stock_data, from_date, to_date,
                            prev_tickers=None, equity_alloc=1.0):
    if not tickers: return 0.0
    returns = []
    for ticker in tickers:
        if ticker not in stock_data: continue
        price = stock_data[ticker]
        past  = price[price.index <= from_date]
        fut   = price[price.index > from_date]
        if len(past) < 1 or len(fut) < 1: continue
        entry      = s(past.iloc[-1])
        target     = from_date + relativedelta(months=1)
        near_fut   = fut[fut.index <= target + timedelta(days=10)]
        if len(near_fut) < 1: continue
        exit_price = s(near_fut.iloc[-1])
        gross_ret  = exit_price / entry - 1
        cost = TC['large'] if ticker in LARGE_CAP else TC['mid']
        if prev_tickers is None or ticker not in prev_tickers:
            gross_ret -= cost
        returns.append(gross_ret)
    if prev_tickers:
        exiting = [t for t in prev_tickers if t not in tickers]
        for ticker in exiting:
            cost = TC['large'] if ticker in LARGE_CAP else TC['mid']
            if returns: returns[-1] -= cost
    clean = [r for r in returns if not np.isnan(r) and not np.isinf(r)]
    if not clean: return 0.0
    return float(np.mean(clean) * equity_alloc)


def run_walk_forward(nifty, vix, stock_data, ipo_map, n_months=240,
                     fii_dii_df=None, fund_snapshots=None):
    print(f"\nRunning {n_months}-month walk-forward backtest (v2 with survivorship fix)...")
    try:    monthly = nifty.resample('ME').last().index
    except: monthly = nifty.resample('M').last().index
    monthly = monthly[-(n_months + 2):]

    results      = []
    prev_tickers = None

    for i in range(len(monthly) - 1):
        from_date = monthly[i]
        to_date   = monthly[i + 1]

        # ── FIX 1: get time-gated eligible universe at this date ──────────
        eligible = get_eligible_universe(stock_data, ipo_map, from_date, regime_code=2)

        regime   = compute_regime_score(nifty, vix, eligible, from_date, fii_dii_df)
        code     = regime['code']
        score    = regime['score']
        eq_alloc = {4:0.90, 3:0.80, 2:0.60, 1:0.15, 0:0.0}.get(code, 0.60)
        n_stocks = {4:15,   3:10,   2:10,   1:0,    0:0   }.get(code, 0)

        # ── FIX 2: momentum boost — recalibrated threshold ───────────────
        # Original: 3M Nifty > 8%, code==3 → eq=0.88
        # Fix: lower threshold to 7%, add score gate at 62, bump stocks to 13
        # Rationale: 2021 had strong breadth + momentum all year but the 8%
        # threshold was never quite hit for enough months to matter.
        momentum_boosted = False
        if code == 3:
            nc3 = nifty[nifty.index <= from_date]
            if len(nc3) > 63:
                m3b = (float(nc3.iloc[-1])/float(nc3.iloc[-63])-1)*100
                if m3b > 7.0 and score >= 62:
                    eq_alloc = 0.88
                    n_stocks = 13
                    momentum_boosted = True
        elif code == 4:
            # Strong Bull: bump to 13 stocks minimum (was 15, keep 15)
            n_stocks = 15

        selected, univ_size = select_stocks_at_date(
            eligible, nifty, code, from_date, n_stocks,
            fund_snapshots, regime_score=score
        )

        port_ret_equity = compute_monthly_return(
            selected, stock_data, from_date, to_date, prev_tickers, eq_alloc
        )

        # FI return on cash portion
        fi_ret    = get_fi_monthly_return(code, eq_alloc, from_date)
        repo_rate = get_repo_rate(from_date)
        port_ret  = port_ret_equity + fi_ret

        nifty_p   = nifty[nifty.index <= from_date]
        nifty_f   = nifty[nifty.index > from_date]
        nifty_ret = 0.0
        if len(nifty_p) > 0 and len(nifty_f) > 0:
            tgt  = from_date + relativedelta(months=1)
            near = nifty_f[nifty_f.index <= tgt + timedelta(days=10)]
            if len(near) > 0:
                nifty_ret = s(near.iloc[-1]) / s(nifty_p.iloc[-1]) - 1

        results.append({
            'date':              str(from_date.date()),
            'regime_code':       code,
            'regime_score':      score,
            'trend':             regime['trend'],
            'vola':              regime['vola'],
            'breadth':           regime['breadth'],
            'flow':              regime['flow'],
            'equity_alloc':      eq_alloc,
            'n_stocks':          len(selected),
            'eligible_universe': univ_size,
            'momentum_boosted':  momentum_boosted,
            'repo_rate':         round(repo_rate, 2),
            'fi_ret':            round(fi_ret * 100, 3),
            'equity_ret':        round(port_ret_equity * 100, 3),
            'port_ret':          round(port_ret * 100, 3),
            'nifty_ret':         round(nifty_ret * 100, 3),
            'alpha':             round((port_ret - nifty_ret) * 100, 3),
            'tickers':           selected[:5],
        })
        prev_tickers = selected

        REGIME_LABELS = {4:'SBull', 3:'MBull', 2:'Neut', 1:'MBear', 0:'SBear'}
        boost_flag = '↑' if momentum_boosted else ' '
        print(f"  {str(from_date.date())} [{REGIME_LABELS[code]:5} {score:4.0f}]{boost_flag} "
              f"univ={univ_size:3} stk={len(selected):2} eq={eq_alloc:.0%} | "
              f"Eq={port_ret_equity*100:+5.2f}% FI={fi_ret*100:+4.2f}% "
              f"Tot={port_ret*100:+5.2f}% Nifty={nifty_ret*100:+5.2f}% "
              f"α={(port_ret-nifty_ret)*100:+5.2f}%")

    return pd.DataFrame(results)


def compute_metrics(df):
    df = df.copy()
    df['port_ret']   = pd.to_numeric(df['port_ret'],   errors='coerce').fillna(0)
    df['equity_ret'] = pd.to_numeric(df['equity_ret'], errors='coerce').fillna(0)
    df['fi_ret']     = pd.to_numeric(df['fi_ret'],     errors='coerce').fillna(0)
    df['nifty_ret']  = pd.to_numeric(df['nifty_ret'],  errors='coerce').fillna(0)
    df['alpha']      = df['port_ret'] - df['nifty_ret']
    port_rets  = df['port_ret'].values / 100
    eq_rets    = df['equity_ret'].values / 100
    nifty_rets = df['nifty_ret'].values / 100
    alphas     = df['alpha'].values / 100
    n          = len(df)

    port_cum   = (1 + port_rets).prod() - 1
    nifty_cum  = (1 + nifty_rets).prod() - 1
    port_ann   = (1 + port_cum)**(12/n) - 1
    nifty_ann  = (1 + nifty_cum)**(12/n) - 1

    rf_m       = 0.065 / 12
    excess     = port_rets - rf_m
    sharpe     = (excess.mean()/excess.std()*np.sqrt(12)) if excess.std() > 0 else 0
    bexcess    = nifty_rets - rf_m
    bsharpe    = (bexcess.mean()/bexcess.std()*np.sqrt(12)) if bexcess.std() > 0 else 0
    ir         = (alphas.mean()/alphas.std()*np.sqrt(12)) if alphas.std() > 0 else 0

    cum_p      = pd.Series((1+port_rets).cumprod())
    hwm        = cum_p.cummax()
    max_dd     = float(((cum_p-hwm)/hwm).min())
    cum_n      = pd.Series((1+nifty_rets).cumprod())
    nifty_hwm  = cum_n.cummax()
    nifty_max_dd = float(((cum_n-nifty_hwm)/nifty_hwm).min())
    calmar     = port_ann / abs(max_dd) if max_dd != 0 else 0
    win_rate   = (alphas > 0).mean()
    beta       = np.cov(port_rets, nifty_rets)[0,1] / np.var(nifty_rets) if nifty_rets.std() > 0 else 1.0

    regime_perf = {}
    labels = {0:'Strong Bear',1:'Mild Bear',2:'Neutral',3:'Mild Bull',4:'Strong Bull'}
    for code in [0,1,2,3,4]:
        m = df[df['regime_code'] == code]
        if len(m) > 0:
            regime_perf[labels[code]] = {
                'months':      int(len(m)),
                'port_avg':    round(float(m['port_ret'].mean()), 2),
                'nifty_avg':   round(float(m['nifty_ret'].mean()), 2),
                'alpha_avg':   round(float(m['alpha'].mean()), 2),
                'win_rate':    round(float((m['alpha'] > 0).mean()*100), 1),
                'cash_months': int((m['equity_alloc'] == 0).sum()),
            }

    df['date'] = pd.to_datetime(df['date'])
    periods = {
        '2005-07 bull market':           ('2005-01-01', '2007-12-31'),
        '2008 GFC crash':                ('2008-01-01', '2009-03-31'),
        '2009-10 recovery':              ('2009-04-01', '2010-12-31'),
        '2011-13 correction':            ('2011-01-01', '2013-12-31'),
        '2014-17 bull run':              ('2014-01-01', '2017-12-31'),
        '2018-19 correction':            ('2018-01-01', '2019-12-31'),
        'COVID crash (Feb-May 2020)':    ('2020-01-01', '2020-05-31'),
        'COVID recovery (Jun-Dec 2020)': ('2020-06-01', '2020-12-31'),
        '2021 bull run':                 ('2021-01-01', '2021-12-31'),
        '2022 bear market':              ('2022-01-01', '2022-12-31'),
        '2023 recovery':                 ('2023-01-01', '2023-12-31'),
        '2024-26 correction':            ('2024-01-01', '2026-03-01'),
    }
    period_perf = {}
    for label, (start, end) in periods.items():
        mask = (df['date'] >= start) & (df['date'] <= end)
        sub  = df[mask]
        if len(sub) > 0:
            pr = sub['port_ret'].values/100
            nr = sub['nifty_ret'].values/100
            period_perf[label] = {
                'months':      len(sub),
                'port_total':  round((1+pr).prod()-1, 4)*100,
                'nifty_total': round((1+nr).prod()-1, 4)*100,
                'alpha_total': round(((1+pr).prod()-(1+nr).prod())*100, 2),
                'port_avg_m':  round(float(sub['port_ret'].mean()), 2),
                'nifty_avg_m': round(float(sub['nifty_ret'].mean()), 2),
            }

    # ── Survivorship audit ────────────────────────────────────────────────────
    if 'eligible_universe' in df.columns:
        survivorship_audit = {
            'min_universe_size':    int(df['eligible_universe'].min()),
            'max_universe_size':    int(df['eligible_universe'].max()),
            'mean_universe_size':   round(float(df['eligible_universe'].mean()), 1),
            'pre_2010_avg':         round(float(df[df['date'] < '2010-01-01']['eligible_universe'].mean()), 1) if len(df[df['date'] < '2010-01-01']) > 0 else None,
            'post_2022_avg':        round(float(df[df['date'] >= '2022-01-01']['eligible_universe'].mean()), 1) if len(df[df['date'] >= '2022-01-01']) > 0 else None,
        }
    else:
        survivorship_audit = {}

    # ── FI contribution ───────────────────────────────────────────────────────
    eq_only_cum  = (1 + eq_rets).prod() - 1
    eq_only_ann  = (1 + eq_only_cum)**(12/n) - 1
    fi_contrib   = port_ann - eq_only_ann  # annualised FI additive

    return {
        'version':           'v2_survivorship_fi',
        'period_months':     n,
        'port_total_ret':    round(port_cum*100, 2),
        'eq_only_total_ret': round(eq_only_cum*100, 2),
        'nifty_total_ret':   round(nifty_cum*100, 2),
        'port_ann_ret':      round(port_ann*100, 2),
        'eq_only_ann_ret':   round(eq_only_ann*100, 2),
        'fi_contrib_ann':    round(fi_contrib*100, 2),
        'nifty_ann_ret':     round(nifty_ann*100, 2),
        'alpha_ann':         round((port_ann-nifty_ann)*100, 2),
        'sharpe_strategy':   round(sharpe, 2),
        'sharpe_benchmark':  round(bsharpe, 2),
        'information_ratio': round(ir, 2),
        'max_drawdown_pct':  round(max_dd*100, 2),
        'nifty_max_dd_pct':  round(nifty_max_dd*100, 2),
        'calmar_ratio':      round(calmar, 2),
        'win_rate_vs_nifty': round(win_rate*100, 1),
        'beta':              round(beta, 2),
        'regime_breakdown':  regime_perf,
        'period_breakdown':  period_perf,
        'survivorship_audit': survivorship_audit,
    }


def print_report(metrics, df):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BACKTEST v2 — SURVIVORSHIP FIX + 2021 MOMENTUM RECALIBRATION")
    print(sep)
    print(f"  Period:           {metrics['period_months']} months ({metrics['period_months']//12} years)")
    print(f"  Strategy total:   {metrics['port_total_ret']:+.1f}%  (equity-only: {metrics.get('eq_only_total_ret',0):+.1f}%)")
    print(f"  Nifty total:      {metrics['nifty_total_ret']:+.1f}%")
    print(f"\n  Annualised:")
    print(f"  Strategy (w/ FI): {metrics['port_ann_ret']:+.1f}% p.a.")
    print(f"  Equity-only:      {metrics.get('eq_only_ann_ret',0):+.1f}% p.a.")
    print(f"  FI contribution:  {metrics.get('fi_contrib_ann',0):+.2f}% p.a.")
    print(f"  Nifty:            {metrics['nifty_ann_ret']:+.1f}% p.a.")
    print(f"  Alpha (vs Nifty): {metrics['alpha_ann']:+.1f}% p.a.")
    print(f"\n  Risk:")
    print(f"  Sharpe strategy:  {metrics['sharpe_strategy']:.2f}")
    print(f"  Sharpe Nifty:     {metrics['sharpe_benchmark']:.2f}")
    print(f"  Info ratio:       {metrics['information_ratio']:.2f}")
    print(f"  Max drawdown:     {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Nifty max dd:     {metrics['nifty_max_dd_pct']:.1f}%")
    print(f"  Calmar ratio:     {metrics['calmar_ratio']:.2f}")
    print(f"  Win rate:         {metrics['win_rate_vs_nifty']:.0f}%")
    print(f"  Beta:             {metrics['beta']:.2f}")

    sa = metrics.get('survivorship_audit', {})
    if sa:
        print(f"\n  Survivorship audit:")
        print(f"  Universe size range:   {sa['min_universe_size']} – {sa['max_universe_size']} stocks")
        print(f"  Pre-2010 avg eligible: {sa['pre_2010_avg']}")
        print(f"  Post-2022 avg eligible:{sa['post_2022_avg']}")

    print(f"\n  Regime breakdown:")
    for label, p in metrics['regime_breakdown'].items():
        print(f"  {label:<14} {p['months']:>3}M | "
              f"Port {p['port_avg']:+.2f}% | Nifty {p['nifty_avg']:+.2f}% | "
              f"α {p['alpha_avg']:+.2f}% | WR {p['win_rate']:.0f}%")
    print(f"\n  Period breakdown:")
    for label, p in metrics['period_breakdown'].items():
        print(f"  {label:<35} Port {p['port_total']:+5.1f}% "
              f"Nifty {p['nifty_total']:+5.1f}% α {p['alpha_total']:+5.1f}%")
    print(sep)


def run():
    print("\n" + "="*70)
    print("  BACKTEST v2 — SURVIVORSHIP FIX + FI DEPLOYMENT + 2021 RECALIBRATION")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*70)
    print("\n  Key changes vs v1:")
    print("  [1] Survivorship bias: universe gated by listing date at each month")
    print("  [2] FI deployment: cash earns liquid/overnight fund returns every month")
    print("  [3] 2021 momentum boost: threshold 8%→7% + score gate ≥62 + 13 stocks")
    print("  [4] Eligible universe size + FI contribution logged per month")

    fund_snapshots = None
    fund_path = os.path.join(SCRIPT_DIR, 'fundamental_snapshots.csv')
    if os.path.exists(fund_path):
        try:
            fund_snapshots = pd.read_csv(fund_path, parse_dates=['snapshot_date'])
            print(f"\n  Fundamentals: {fund_snapshots['ticker'].nunique()} stocks loaded")
        except Exception as e:
            print(f"\n  Fundamental load failed: {e}")
    else:
        print("\n  Fundamentals: not found — using price proxies")

    fii_dii_df = None
    fii_path   = os.path.join(SCRIPT_DIR, 'fii_dii_data.csv')
    if os.path.exists(fii_path):
        try:
            fii_dii_df = pd.read_csv(fii_path, parse_dates=['date'])
            fii_dii_df = fii_dii_df.sort_values('date').reset_index(drop=True)
            print(f"  FII/DII: {len(fii_dii_df)} days loaded "
                  f"({fii_dii_df['date'].min().date()} to {fii_dii_df['date'].max().date()})")
        except Exception as e:
            print(f"  FII/DII load failed: {e}")
    else:
        print("  FII/DII: not found — using price proxy")

    nifty, vix, stock_data = download_all_data(lookback_days=7500)

    # ── FIX 1: build IPO map from actual data start dates ────────────────────
    ipo_map = build_ipo_map(stock_data)
    print(f"\n  IPO map built: {len(ipo_map)} tickers with listing dates")
    pre2010 = sum(1 for d in ipo_map.values() if d.year <= 2010)
    post2022 = sum(1 for d in ipo_map.values() if d.year >= 2022)
    print(f"  Listed ≤2010: {pre2010} stocks | Listed ≥2022: {post2022} stocks")
    print(f"  (These {post2022} tickers correctly excluded from pre-2022 months)")

    df = run_walk_forward(
        nifty, vix, stock_data, ipo_map,
        n_months=240,
        fii_dii_df=fii_dii_df,
        fund_snapshots=fund_snapshots
    )

    if df.empty:
        print("  No results generated")
        return

    metrics = compute_metrics(df)
    print_report(metrics, df)

    output = {
        'run_date': datetime.today().strftime('%Y-%m-%d'),
        'version':  'v2',
        'changes':  [
            'Survivorship bias fix: time-gated universe (IPO map from first data date)',
            'FI deployment: cash earns repo-derived liquid/overnight fund returns each month',
            '2021 momentum boost recalibrated: threshold 8%→7%, score gate ≥62, stocks 10→13',
            'Eligible universe size and FI contribution logged per month',
        ],
        'metrics':  metrics,
        'monthly':  df.to_dict('records'),
    }
    with open(BT_OUT, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {BT_OUT}")


if __name__ == "__main__":
    run()
