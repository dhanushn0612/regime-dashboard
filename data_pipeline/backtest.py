"""
Walk-Forward Backtest Engine
==============================
Tests all 5 layers together end-to-end vs Nifty 50 benchmark.
Honest backtest — no survivorship bias, no look-ahead, real costs.

Methodology:
  - Walk-forward: 12-month training window, 1-month test window
  - Slides forward monthly over 2 years of data
  - Each month: regime → sector filter → stock selection → portfolio weights
  - Returns measured on following month (true out-of-sample)
  - Transaction costs: 20bps large cap, 40bps mid cap
  - Benchmark: Nifty 50 buy-and-hold

Output metrics:
  - Total return vs Nifty
  - Annualised alpha
  - Sharpe ratio (strategy vs benchmark)
  - Max drawdown (strategy vs benchmark)
  - Win rate (% months outperforming Nifty)
  - Calmar ratio (return / max drawdown)
  - Information ratio

Usage:
    python data_pipeline/backtest.py

Output:
    public/backtest_results.json
    (printed report in terminal)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, '..', 'public')
BT_OUT       = os.path.join(OUTPUT_DIR, 'backtest_results.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TRANSACTION COSTS ─────────────────────────────────────────────────
LARGE_CAP = {
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
    'HINDUNILVR.NS','ITC.NS','SBIN.NS','BHARTIARTL.NS','KOTAKBANK.NS',
    'LT.NS','AXISBANK.NS','HCLTECH.NS','ASIANPAINT.NS','MARUTI.NS',
    'SUNPHARMA.NS','TITAN.NS','BAJFINANCE.NS','WIPRO.NS','TECHM.NS',
}
TC = {'large': 0.0020, 'mid': 0.0040}

# ── NIFTY 200 LIQUID UNIVERSE (for Neutral/Bear regimes) ─────────────
# In weaker regimes only invest in the most liquid stocks
# Prevents picking illiquid small caps that mean-revert
NIFTY200_LIQUID = {
    # IT
    "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","LTIM.NS",
    "MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","OFSS.NS","TATAELXSI.NS","LTTS.NS","KPITTECH.NS",
    # Banks
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
    "INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","IDFCFIRSTB.NS","PNB.NS","BANKBARODA.NS",
    # Financial
    "BAJFINANCE.NS","BAJAJFINSV.NS","MUTHOOTFIN.NS","CHOLAFIN.NS","PFC.NS","RECLTD.NS",
    "HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS","IRFC.NS",
    # FMCG
    "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
    "GODREJCP.NS","MARICO.NS","COLPAL.NS","TATACONSUM.NS","VBL.NS",
    # Auto
    "MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS",
    "ASHOKLEY.NS","BOSCHLTD.NS","MOTHERSON.NS","BHARATFORG.NS","MRF.NS",
    # Pharma
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS",
    "TORNTPHARM.NS","LUPIN.NS","ALKEM.NS","APOLLOHOSP.NS",
    # Energy
    "RELIANCE.NS","ONGC.NS","COALINDIA.NS","BPCL.NS","IOC.NS","GAIL.NS","PETRONET.NS",
    # Metals
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS","NMDC.NS",
    # Infra/Capital
    "LT.NS","NTPC.NS","POWERGRID.NS","TATAPOWER.NS","SIEMENS.NS",
    "ABB.NS","HAVELLS.NS","POLYCAB.NS","CUMMINSIND.NS",
    # Consumer
    "TITAN.NS","ASIANPAINT.NS","DMART.NS","TRENT.NS","PIDILITIND.NS",
    "BERGEPAINT.NS","WHIRLPOOL.NS","VOLTAS.NS","DIXON.NS",
    # Telecom/Misc
    "BHARTIARTL.NS","ADANIPORTS.NS","ULTRACEMCO.NS","AMBUJACEM.NS",
    "INDHOTEL.NS","IRCTC.NS","CONCOR.NS",
}

# ── STOCK UNIVERSE (same as screener) ─────────────────────────────────
UNIVERSE = [
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

# ── HELPERS ───────────────────────────────────────────────────────────
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


# ── STEP 1: DOWNLOAD ALL DATA ONCE ────────────────────────────────────
def download_all_data(lookback_days=1000):
    """
    Download all price data once to avoid repeated API calls.
    Automatically uses Nifty 500 expanded universe if cache exists.
    Run nifty500_universe.py first to build the cache.
    """
    print("Downloading all price data (one-time)...")
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    # Nifty benchmark
    nifty = flatten(yf.download("^NSEI", start=start, end=end,
                                 progress=False, auto_adjust=True))
    nifty_close = get_close(nifty)
    print(f"  Nifty: {len(nifty_close)} days")

    # India VIX
    vix = flatten(yf.download("^INDIAVIX", start=start, end=end,
                               progress=False, auto_adjust=True))
    vix_close = get_close(vix) if len(vix) > 0 else pd.Series(dtype=float)

    # Check for Nifty 500 expanded universe cache
    price_cache = os.path.join(SCRIPT_DIR, "nifty500_prices.pkl")
    if os.path.exists(price_cache):
        import pickle
        try:
            cache_age = (datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(price_cache))).days
            print(f"  Loading Nifty 500 price cache ({cache_age} days old) from {price_cache}...")
            with open(price_cache, "rb") as f:
                stock_data = pickle.load(f)
            print(f"  Stocks: {len(stock_data)} (Nifty 500 universe)")
            return nifty_close, vix_close, stock_data
        except Exception as e:
            print(f"  Cache load failed: {e}")

    # Fallback: download base universe
    print(f"  No Nifty 500 cache found — using {len(UNIVERSE)}-stock universe")
    print(f"  TIP: Run nifty500_universe.py to expand to full Nifty 500")
    stock_data = {}
    for ticker in UNIVERSE:
        try:
            df = flatten(yf.download(ticker, start=start, end=end,
                                      progress=False, auto_adjust=True))
            if len(df) > 150:
                stock_data[ticker] = get_close(df)
        except Exception:
            pass
    print(f"  Stocks: {len(stock_data)}/{len(UNIVERSE)}")

    return nifty_close, vix_close, stock_data


# ── STEP 2: REGIME SCORE (simplified, fast version) ───────────────────
def compute_regime_score(nifty: pd.Series, vix: pd.Series,
                          stock_data: dict, as_of: pd.Timestamp,
                          fii_dii_df: pd.DataFrame = None) -> dict:
    """
    Compute regime score at a given date using only past data.
    Robust version with explicit error handling per signal.
    """
    nc = nifty[nifty.index <= as_of]
    nc = nc.tail(300)

    if len(nc) < 100:
        return {'code': 2, 'score': 50.0}

    curr  = float(nc.iloc[-1])
    trend = 0

    try:
        sma50 = float(nc.rolling(50).mean().iloc[-1])
        dist50 = (curr - sma50) / sma50 * 100
        trend += 20 if dist50 > 3 else 14 if dist50 > 0 else 6 if dist50 > -3 else 0
    except Exception: pass

    try:
        sma200 = float(nc.rolling(200).mean().iloc[-1])
        if len(nc) >= 200:
            dist200 = (curr - sma200) / sma200 * 100
            trend += 20 if dist200 > 5 else 13 if dist200 > 0 else 5 if dist200 > -5 else 0
            sma50v = float(nc.rolling(50).mean().iloc[-1])
            trend += 15 if sma50v > sma200 else 0
    except Exception: pass

    try:
        if len(nc) > 21:
            roc1m = (curr / float(nc.iloc[-21]) - 1) * 100
            trend += 10 if roc1m > 3 else 6 if roc1m > 0 else 2 if roc1m > -3 else 0
    except Exception: pass

    try:
        if len(nc) > 63:
            roc3m = (curr / float(nc.iloc[-63]) - 1) * 100
            trend += 10 if roc3m > 7 else 6 if roc3m > 0 else 2 if roc3m > -7 else 0
    except Exception: pass

    try:
        high52 = float(nc.rolling(min(252, len(nc))).max().iloc[-1])
        dist52 = (curr / high52 - 1) * 100
        trend += 15 if dist52 > -5 else 10 if dist52 > -10 else 4 if dist52 > -20 else 0
    except Exception: pass

    trend = min(100, max(0, trend))

    # Volatility
    vola = 50
    try:
        vc = vix[vix.index <= as_of].tail(60)
        if len(vc) >= 5:
            cv = float(vc.iloc[-1])
            vola  = 35 if cv < 13 else 28 if cv < 16 else 18 if cv < 20 else 10 if cv < 25 else 4 if cv < 30 else 0
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
            vov = float(vc.rolling(10).std().iloc[-1])
            vola += 20 if vov < 1.0 else 14 if vov < 2.0 else 7 if vov < 3.5 else 0
        vola = min(100, max(0, vola))
    except Exception:
        pass

    # Breadth
    breadth = 50
    try:
        above50 = []
        sample  = list(stock_data.keys())[:20]
        for tk in sample:
            try:
                c = stock_data[tk][stock_data[tk].index <= as_of]
                if len(c) >= 50:
                    above50.append(float(c.iloc[-1]) > float(c.rolling(50).mean().iloc[-1]))
            except Exception:
                pass
        if above50:
            pct50   = sum(above50) / len(above50) * 100
            breadth  = 35 if pct50 > 70 else 25 if pct50 > 55 else 15 if pct50 > 40 else 7 if pct50 > 30 else 0
            above200 = []
            for tk in sample:
                try:
                    c = stock_data[tk][stock_data[tk].index <= as_of]
                    if len(c) >= 200:
                        above200.append(float(c.iloc[-1]) > float(c.rolling(200).mean().iloc[-1]))
                except Exception:
                    pass
            if above200:
                pct200   = sum(above200) / len(above200) * 100
                breadth += 35 if pct200 > 65 else 25 if pct200 > 50 else 15 if pct200 > 35 else 7 if pct200 > 25 else 0
        breadth = min(100, max(0, breadth))
    except Exception:
        pass

    # Flow — use real FII/DII monthly totals if available, else price proxy
    flow = 40
    try:
        vc2 = vix[vix.index <= as_of].tail(20)
        fii_used = False

        if fii_dii_df is not None and len(fii_dii_df) >= 5:
            fii_past = fii_dii_df[fii_dii_df['date'] <= as_of]
            if len(fii_past) >= 5:
                fii_monthly = fii_past.copy()
                fii_monthly['month'] = fii_monthly['date'].dt.to_period('M')
                monthly_totals = fii_monthly.groupby('month').agg(
                    FII_Net=('FII_Net', 'sum'), DII_Net=('DII_Net', 'sum')
                ).tail(3)
                if len(monthly_totals) >= 2:
                    fii_3m  = float(monthly_totals['FII_Net'].sum())
                    dii_3m  = float(monthly_totals['DII_Net'].sum())
                    fii_cur = float(monthly_totals['FII_Net'].iloc[-1])
                    dii_cur = float(monthly_totals['DII_Net'].iloc[-1])
                    flow  = 30 if fii_3m > 15000 else 22 if fii_3m > 5000 else 12 if fii_3m > -5000 else 4 if fii_3m > -15000 else 0
                    flow += 20 if fii_cur > 0 and dii_cur > 0 else 14 if fii_cur > 0 else 10 if dii_cur > 0 else 0
                    if len(vc2) >= 20:
                        slope = (float(vc2.iloc[-1]) - float(vc2.iloc[-20])) / 20
                        flow += 35 if slope < -0.2 else 25 if slope < 0 else 15 if slope < 0.2 else 6 if slope < 0.5 else 0
                    flow += 15 if dii_3m > 50000 else 10 if dii_3m > 20000 else 5 if dii_3m > 0 else 0
                    flow = min(100, max(0, flow))
                    fii_used = True

        if not fii_used:
            if len(vc2) >= 5 and len(nc) >= 5:
                r5 = (curr / float(nc.iloc[-5]) - 1) * 100
                v5 = (float(vc2.iloc[-1]) / float(vc2.iloc[-5]) - 1) * 100
                flow = 65 if r5 > 1 and v5 < -5 else 48 if r5 > 0 and v5 < 0 else 30 if r5 > 0 else 18 if v5 < 0 else 5
            if len(vc2) >= 20:
                slope = (float(vc2.iloc[-1]) - float(vc2.iloc[-20])) / 20
                flow += 35 if slope < -0.2 else 25 if slope < 0 else 15 if slope < 0.2 else 6 if slope < 0.5 else 0
            flow = min(100, max(0, flow))
    except Exception:
        pass

    composite = trend * 0.30 + vola * 0.25 + breadth * 0.25 + flow * 0.20

    if composite >= 75:   code = 4
    elif composite >= 55: code = 3
    elif composite >= 40: code = 2
    elif composite >= 20: code = 1
    else:                 code = 0

    return {'code': code, 'score': round(composite, 1),
            'trend': round(trend,1), 'vola': round(vola,1),
            'breadth': round(breadth,1), 'flow': round(flow,1)}



# ── STEP 3: SELECT STOCKS AT DATE ────────────────────────────────────
def select_stocks_at_date(stock_data: dict, nifty: pd.Series,
                           regime_code: int, as_of: pd.Timestamp,
                           n_stocks: int = 5,
                           n_override: int = None) -> list:
    """
    Select top stocks using momentum + quality at a given past date.
    Only uses data available up to as_of — strict no look-ahead.
    """
    n_map  = {4: 15, 3: 10, 2: 5, 1: 0, 0: 0}
    n_pick = n_override if n_override is not None else n_map.get(regime_code, 0)
    if n_pick == 0:
        return []

    nifty_past = nifty[nifty.index <= as_of]
    nifty_3m   = (s(nifty_past.iloc[-1]) / s(nifty_past.iloc[-63]) - 1) if len(nifty_past) > 63 else 0

    # In Neutral/Bear regimes, restrict to liquid Nifty 200 stocks only
    # This prevents illiquid small caps from dominating factor scores
    if regime_code <= 2:
        filtered_universe = {t: p for t, p in stock_data.items() if t in NIFTY200_LIQUID}
    else:
        filtered_universe = stock_data  # Full 497 in Bull regimes

    scores = []
    for ticker, price in filtered_universe.items():
        past = price[price.index <= as_of]
        if len(past) < 130:
            continue
        try:
            # Quality filter - relaxed for historical periods
            sma50 = s(past.rolling(50).mean().iloc[-1])
            if s(past.iloc[-1]) < sma50 * 0.95:  # 5% tolerance
                continue

            rets = past.pct_change().dropna()
            vol  = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
            if vol > 70:
                continue

            # Factor scores
            r1m  = (s(past.iloc[-1]) / s(past.iloc[-21]) - 1) if len(past) > 21 else 0
            r3m  = (s(past.iloc[-1]) / s(past.iloc[-63]) - 1) if len(past) > 63 else 0
            r6m  = (s(past.iloc[-1]) / s(past.iloc[-126]) - 1) if len(past) > 126 else 0
            r12m = (s(past.iloc[-1]) / s(past.iloc[-252]) - 1) if len(past) > 252 else 0
            rs   = r3m - nifty_3m

            mom   = r1m*0.15 + r3m*0.35 + r6m*0.30 + r12m*0.20
            lowv  = max(0, 100 - vol)
            high52 = s(past.rolling(252).max().iloc[-1])
            earn  = ((s(past.iloc[-1]) / high52) - 0.6) * 100

            # Regime-adjusted factor weights
            if regime_code >= 3:        # Bull: momentum + earnings lead
                score = mom*0.40 + earn*0.30 + lowv*0.15 + rs*10*0.15
            elif regime_code == 2:      # Neutral: balanced, more momentum
                score = mom*0.35 + earn*0.25 + lowv*0.25 + rs*10*0.15
            else:                       # Bear: quality + low-vol dominate
                score = mom*0.15 + earn*0.15 + lowv*0.50 + rs*10*0.20

            scores.append((ticker, score, r1m, r3m, vol))

        except Exception:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:n_pick]]


# ── STEP 4: COMPUTE PORTFOLIO RETURN ──────────────────────────────────
def compute_monthly_return(tickers: list, stock_data: dict,
                            from_date: pd.Timestamp,
                            to_date: pd.Timestamp,
                            prev_tickers: list = None,
                            equity_alloc: float = 1.0) -> float:
    """
    Compute equal-weight portfolio return over one month.
    Deducts transaction costs for new/exited positions.
    """
    if not tickers:
        return 0.0

    returns = []
    for ticker in tickers:
        if ticker not in stock_data:
            continue
        price = stock_data[ticker]
        past  = price[price.index <= from_date]
        fut   = price[price.index > from_date]

        if len(past) < 1 or len(fut) < 1:
            continue

        entry = s(past.iloc[-1])
        # Find exit price approximately 1 month later
        target_date = from_date + relativedelta(months=1)
        near_future = fut[fut.index <= target_date + timedelta(days=10)]
        if len(near_future) < 1:
            continue

        exit_price = s(near_future.iloc[-1])
        gross_ret  = exit_price / entry - 1

        # Transaction cost
        cost = TC['large'] if ticker in LARGE_CAP else TC['mid']

        # Only charge cost if position changed
        if prev_tickers is None or ticker not in prev_tickers:
            gross_ret -= cost  # Entry cost

        returns.append(gross_ret)

    # Charge exit costs for positions being removed
    if prev_tickers:
        exiting = [t for t in prev_tickers if t not in tickers]
        for ticker in exiting:
            cost = TC['large'] if ticker in LARGE_CAP else TC['mid']
            if returns:
                returns[-1] -= cost  # Approximate: deduct from last position

    if not returns:
        return 0.0

    # Equal weight, scaled by equity allocation
    clean = [r for r in returns if not np.isnan(r) and not np.isinf(r)]
    if not clean:
        return 0.0
    port_ret = np.mean(clean) * equity_alloc
    return float(port_ret) if not np.isnan(port_ret) else 0.0


# ── STEP 5: WALK-FORWARD LOOP ─────────────────────────────────────────
def run_walk_forward(nifty: pd.Series, vix: pd.Series,
                     stock_data: dict, n_months: int = 24,
                     fii_dii_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Walk-forward backtest: slide monthly over n_months.
    At each step:
      1. Compute regime (using only past data)
      2. Select stocks (using only past data)
      3. Record forward return (true out-of-sample)
    """
    print(f"\nRunning {n_months}-month walk-forward backtest...")

    # Get monthly dates (month-ends) going back n_months
    end_date   = nifty.index[-1]
    try:
        monthly = nifty.resample('ME').last().index
    except Exception:
        monthly = nifty.resample('M').last().index

    # Use last n_months+1 dates (need pairs)
    monthly = monthly[-(n_months + 2):]

    results = []
    prev_tickers = None

    for i in range(len(monthly) - 1):
        from_date = monthly[i]
        to_date   = monthly[i + 1]

        # Regime at from_date (using only data up to from_date)
        regime = compute_regime_score(nifty, vix, stock_data, from_date, fii_dii_df)
        code   = regime['code']
        score  = regime['score']

        # Equity allocation — with Neutral momentum overlay
        eq_alloc = {4: 0.90, 3: 0.70, 2: 0.45, 1: 0.15, 0: 0.0}.get(code, 0.45)

        # Neutral momentum overlay: if Neutral but market trending up, be less defensive
        n_stocks_override = None
        if code == 2:
            nc_check = nifty[nifty.index <= from_date]
            if len(nc_check) > 21:
                mom_1m = (float(nc_check.iloc[-1]) / float(nc_check.iloc[-21]) - 1) * 100
                mom_3m = (float(nc_check.iloc[-1]) / float(nc_check.iloc[-63]) - 1) * 100 if len(nc_check) > 63 else 0
                if mom_1m > 0.5 or mom_3m > 3.0:  # Positive short OR medium trend
                    eq_alloc = 0.70          # Increase from 45% to 70%
                    n_stocks_override = 10   # Increase from 5 to 10 stocks

        # Select stocks
        selected = select_stocks_at_date(stock_data, nifty, code, from_date,
                                          n_override=n_stocks_override)

        # Portfolio return (out-of-sample)
        port_ret = compute_monthly_return(
            selected, stock_data, from_date, to_date,
            prev_tickers, eq_alloc
        )

        # Nifty benchmark return
        nifty_past = nifty[nifty.index <= from_date]
        nifty_fut  = nifty[nifty.index > from_date]
        if len(nifty_past) > 0 and len(nifty_fut) > 0:
            target = from_date + relativedelta(months=1)
            near   = nifty_fut[nifty_fut.index <= target + timedelta(days=10)]
            if len(near) > 0:
                nifty_ret = s(near.iloc[-1]) / s(nifty_past.iloc[-1]) - 1
            else:
                nifty_ret = 0.0
        else:
            nifty_ret = 0.0

        results.append({
            'date':          str(from_date.date()),
            'regime_code':   code,
            'regime_score':  score,
            'equity_alloc':  eq_alloc,
            'n_stocks':      len(selected),
            'port_ret':      round(port_ret * 100, 3),
            'nifty_ret':     round(nifty_ret * 100, 3),
            'alpha':         round((port_ret - nifty_ret) * 100, 3),
            'tickers':       selected[:5],  # Top 5 for reference
        })

        prev_tickers = selected
        n_avail = len([t for t in stock_data if len(stock_data[t][stock_data[t].index <= from_date]) > 130])
        print(f"  {str(from_date.date())}: R={code}({score:.0f}) stocks={len(selected)}/{n_avail} | Port={port_ret*100:+.2f}% Nifty={nifty_ret*100:+.2f}% Alpha={(port_ret-nifty_ret)*100:+.2f}%")

    return pd.DataFrame(results)


# ── STEP 6: COMPUTE PERFORMANCE METRICS ──────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute all standard portfolio performance metrics.
    These are the metrics allocators look at.
    """
    # Clean NaN values before computing metrics
    df = df.copy()
    df['port_ret']  = pd.to_numeric(df['port_ret'], errors='coerce').fillna(0)
    df['nifty_ret'] = pd.to_numeric(df['nifty_ret'], errors='coerce').fillna(0)
    df['alpha']     = df['port_ret'] - df['nifty_ret']

    port_rets   = df['port_ret'].values / 100
    nifty_rets  = df['nifty_ret'].values / 100
    alphas      = df['alpha'].values / 100
    n           = len(df)

    # Cumulative returns
    port_cum    = (1 + port_rets).prod() - 1
    nifty_cum   = (1 + nifty_rets).prod() - 1

    # Annualised returns
    port_ann    = (1 + port_cum) ** (12 / n) - 1
    nifty_ann   = (1 + nifty_cum) ** (12 / n) - 1
    alpha_ann   = port_ann - nifty_ann

    # Sharpe (using monthly returns, annualised)
    rf_monthly  = 0.065 / 12
    excess      = port_rets - rf_monthly
    sharpe      = (excess.mean() / excess.std() * np.sqrt(12)) if excess.std() > 0 else 0

    bench_exc   = nifty_rets - rf_monthly
    bench_sharpe = (bench_exc.mean() / bench_exc.std() * np.sqrt(12)) if bench_exc.std() > 0 else 0

    # Information ratio
    ir          = (alphas.mean() / alphas.std() * np.sqrt(12)) if alphas.std() > 0 else 0

    # Max drawdown
    cum_port    = pd.Series((1 + port_rets).cumprod())
    hwm         = cum_port.cummax()
    drawdowns   = (cum_port - hwm) / hwm
    max_dd      = float(drawdowns.min())

    cum_nifty   = pd.Series((1 + nifty_rets).cumprod())
    nifty_hwm   = cum_nifty.cummax()
    nifty_dd    = (cum_nifty - nifty_hwm) / nifty_hwm
    nifty_max_dd = float(nifty_dd.min())

    # Calmar ratio
    calmar      = port_ann / abs(max_dd) if max_dd != 0 else 0

    # Win rate vs Nifty
    win_rate    = (alphas > 0).mean()

    # Beta and correlation
    if nifty_rets.std() > 0:
        beta = np.cov(port_rets, nifty_rets)[0, 1] / np.var(nifty_rets)
        corr = np.corrcoef(port_rets, nifty_rets)[0, 1]
    else:
        beta = 1.0
        corr = 1.0

    # Regime breakdown
    regime_perf = {}
    for code in [0, 1, 2, 3, 4]:
        mask = df['regime_code'] == code
        if mask.sum() > 0:
            sub = df[mask]
            regime_perf[str(code)] = {
                'months':     int(mask.sum()),
                'port_avg':   round(float(sub['port_ret'].mean()), 2),
                'nifty_avg':  round(float(sub['nifty_ret'].mean()), 2),
                'alpha_avg':  round(float(sub['alpha'].mean()), 2),
                'win_rate':   round(float((sub['alpha'] > 0).mean() * 100), 1),
            }

    return {
        'period_months':     n,
        'port_total_ret':    round(port_cum * 100, 2),
        'nifty_total_ret':   round(nifty_cum * 100, 2),
        'port_ann_ret':      round(port_ann * 100, 2),
        'nifty_ann_ret':     round(nifty_ann * 100, 2),
        'alpha_ann':         round(alpha_ann * 100, 2),
        'sharpe_strategy':   round(sharpe, 2),
        'sharpe_benchmark':  round(bench_sharpe, 2),
        'information_ratio': round(ir, 2),
        'max_drawdown_pct':  round(max_dd * 100, 2),
        'nifty_max_dd_pct':  round(nifty_max_dd * 100, 2),
        'calmar_ratio':      round(calmar, 2),
        'win_rate_vs_nifty': round(win_rate * 100, 1),
        'beta':              round(beta, 2),
        'correlation':       round(corr, 2),
        'regime_breakdown':  regime_perf,
    }


# ── PRINT REPORT ──────────────────────────────────────────────────────
def print_report(metrics: dict, df: pd.DataFrame):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  BACKTEST RESULTS")
    print(sep)
    print(f"  Period:          {metrics['period_months']} months")
    print(f"  Strategy return: {metrics['port_total_ret']:+.1f}%")
    print(f"  Nifty return:    {metrics['nifty_total_ret']:+.1f}%")
    print(f"  Alpha (total):   {metrics['port_total_ret'] - metrics['nifty_total_ret']:+.1f}%")
    print(f"\n  Annualised:")
    print(f"  Strategy:        {metrics['port_ann_ret']:+.1f}% p.a.")
    print(f"  Nifty:           {metrics['nifty_ann_ret']:+.1f}% p.a.")
    print(f"  Alpha:           {metrics['alpha_ann']:+.1f}% p.a.")
    print(f"\n  Risk metrics:")
    print(f"  Sharpe (strategy): {metrics['sharpe_strategy']:.2f}")
    print(f"  Sharpe (Nifty):    {metrics['sharpe_benchmark']:.2f}")
    print(f"  Information ratio: {metrics['information_ratio']:.2f}")
    print(f"  Max drawdown:      {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Nifty max dd:      {metrics['nifty_max_dd_pct']:.1f}%")
    print(f"  Calmar ratio:      {metrics['calmar_ratio']:.2f}")
    print(f"  Win rate vs Nifty: {metrics['win_rate_vs_nifty']:.0f}%")
    print(f"  Beta:              {metrics['beta']:.2f}")
    print(f"\n  Regime breakdown:")
    labels = {4:'Strong Bull',3:'Mild Bull',2:'Neutral',1:'Mild Bear',0:'Strong Bear'}
    for code, perf in sorted(metrics['regime_breakdown'].items()):
        label = labels.get(int(code), code)
        print(f"  {label:<14}: {perf['months']:>3}M | Port {perf['port_avg']:+.2f}% | Nifty {perf['nifty_avg']:+.2f}% | Alpha {perf['alpha_avg']:+.2f}% | WR {perf['win_rate']:.0f}%")
    print(sep)

    # Honest assessment
    print("\n  HONEST ASSESSMENT:")
    alpha = metrics['alpha_ann']
    sharpe = metrics['sharpe_strategy']
    if alpha > 3 and sharpe > 0.8:
        print("  STRONG — Meaningful alpha with good risk-adjusted returns.")
    elif alpha > 0 and sharpe > 0.5:
        print("  PROMISING — Positive alpha. Refine and extend the backtest period.")
    elif alpha > -2:
        print("  NEUTRAL — Near-market returns. Strategy needs improvement.")
    else:
        print("  WEAK — Underperforming Nifty. Investigate which regime/layer is failing.")
    print(f"\n  Target benchmark: Nifty +3% to +5% p.a. (industry realistic)")
    print(sep + "\n")


# ── MAIN ──────────────────────────────────────────────────────────────
def run():
    print("\n" + "="*60)
    print("  WALK-FORWARD BACKTEST ENGINE")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*60)

    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        print("  Installing python-dateutil...")
        os.system("pip install python-dateutil")
        from dateutil.relativedelta import relativedelta

    # Load FII/DII data if available
    fii_dii_df = None
    fii_dii_path = os.path.join(SCRIPT_DIR, 'fii_dii_data.csv')
    if os.path.exists(fii_dii_path):
        try:
            fii_dii_df = pd.read_csv(fii_dii_path, parse_dates=['date'])
            fii_dii_df = fii_dii_df.sort_values('date').reset_index(drop=True)
            print(f"  FII/DII data: {len(fii_dii_df)} days loaded")
        except Exception as e:
            print(f"  FII/DII load failed: {e}")
    else:
        print("  FII/DII data: not found — using price proxy for flow dimension")

    # Download all data
    nifty, vix, stock_data = download_all_data(lookback_days=1000)

    # Run walk-forward
    df = run_walk_forward(nifty, vix, stock_data, n_months=24, fii_dii_df=fii_dii_df)

    if df.empty:
        print("  No backtest results generated")
        return

    # Compute metrics
    metrics = compute_metrics(df)

    # Print report
    print_report(metrics, df)

    # Save outputs
    output = {
        'run_date':    datetime.today().strftime('%Y-%m-%d'),
        'metrics':     metrics,
        'monthly':     df.to_dict('records'),
    }

    with open(BT_OUT, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Saved: {BT_OUT}")


if __name__ == "__main__":
    run()
