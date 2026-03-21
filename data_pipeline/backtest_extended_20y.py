"""
Extended Walk-Forward Backtest — Jan 2019 to Feb 2026
=======================================================
Full 7-year backtest using real FII/DII monthly data.
Covers: 2019 bull, COVID crash (Mar 2020), 2021 bull,
        2022 bear, 2023 recovery, 2024-26 correction.

FIXES vs previous version:
  1. Minimum history reduced 130 → 100 days
  2. SMA floor is regime-dependent: 90% in Bull, 95% in Bear/Neutral
  3. Volatility cap is regime-dependent: 60-75% by regime (was hardcoded 70%)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'public')
BT_OUT     = os.path.join(OUTPUT_DIR, 'backtest_results.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LARGE_CAP = {
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
    'HINDUNILVR.NS','ITC.NS','SBIN.NS','BHARTIARTL.NS','KOTAKBANK.NS',
    'LT.NS','AXISBANK.NS','HCLTECH.NS','ASIANPAINT.NS','MARUTI.NS',
    'SUNPHARMA.NS','TITAN.NS','BAJFINANCE.NS','WIPRO.NS','TECHM.NS',
}
TC = {'large': 0.0020, 'mid': 0.0040}

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

def download_all_data(lookback_days=7500):
    print("Downloading all price data (covers ~20 years)...")
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    nifty = flatten(yf.download("^NSEI", start=start, end=end,
                                 progress=False, auto_adjust=True))
    nifty_close = get_close(nifty)
    print(f"  Nifty: {len(nifty_close)} days ({nifty_close.index[0].date()} to {nifty_close.index[-1].date()})")

    vix = flatten(yf.download("^INDIAVIX", start=start, end=end,
                               progress=False, auto_adjust=True))
    vix_close = get_close(vix) if len(vix) > 0 else pd.Series(dtype=float)
    print(f"  VIX:   {len(vix_close)} days")

    # Step 1: Download core 48 stocks with full 20-year history
    print(f"  Downloading {len(UNIVERSE)} core stocks with 20-year history...")
    stock_data = {}
    for ticker in UNIVERSE:
        try:
            df = flatten(yf.download(ticker, start=start, end=end,
                                      progress=False, auto_adjust=True))
            if len(df) > 150:
                stock_data[ticker] = get_close(df)
        except Exception:
            pass
    print(f"  Core stocks: {len(stock_data)}/{len(UNIVERSE)}")

    # Step 2: Augment with Nifty 500 cache for expanded recent universe
    # The cache has ~2 years of 497 stocks — use it to add extra stocks
    # for the recent period without re-downloading
    nifty500_cache = os.path.join(SCRIPT_DIR, "nifty500_prices.pkl")
    if os.path.exists(nifty500_cache):
        try:
            import pickle
            cache_age = (datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(nifty500_cache))).days
            print(f"  Loading Nifty 500 cache ({cache_age} days old)...")
            with open(nifty500_cache, "rb") as f:
                cached = pickle.load(f)
            added = 0
            for ticker, prices in cached.items():
                if ticker not in stock_data:
                    stock_data[ticker] = prices
                    added += 1
                # For existing tickers: keep the longer history (already downloaded)
            print(f"  Added {added} extra tickers from Nifty 500 cache")
        except Exception as e:
            print(f"  Cache load failed: {e}")

    print(f"  Total universe: {len(stock_data)} stocks")
    print(f"  Note: Extra 450 Nifty 500 stocks only used for post-2022 months")
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
    return {'code': code, 'score': round(composite,1),
            'trend': round(trend,1), 'vola': round(vola,1),
            'breadth': round(breadth,1), 'flow': round(flow,1)}


def select_stocks_at_date(stock_data, nifty, regime_code, as_of, n_override=None, fund_snapshots=None, regime_score=50):
    n_map  = {4:15, 3:10, 2:10, 1:0, 0:0}
    n_pick = n_override if n_override is not None else n_map.get(regime_code, 0)
    if n_pick == 0: return []

    nifty_past = nifty[nifty.index <= as_of]
    nifty_3m   = (s(nifty_past.iloc[-1])/s(nifty_past.iloc[-63])-1) if len(nifty_past) > 63 else 0

    # For recent periods (2022+) use full Nifty 500 universe
    # For older periods use Nifty 200 liquid (core 48 stocks have long history)
    has_nifty500 = len(stock_data) > 200  # Proxy for whether cache is loaded
    if has_nifty500 and regime_code <= 2:
        # Neutral/Bear: use liquid filter regardless of period
        universe = {t:p for t,p in stock_data.items()
                    if t in NIFTY200_LIQUID and len(p[p.index <= as_of]) >= 100}
    elif has_nifty500:
        # Bull with full universe: only use stocks with sufficient history at this date
        universe = {t:p for t,p in stock_data.items()
                    if len(p[p.index <= as_of]) >= 100}
    else:
        universe = {t:p for t,p in stock_data.items()
                    if t in NIFTY200_LIQUID} if regime_code <= 2 else stock_data

    # ── FIX 1: regime-dependent SMA floor ─────────────────────────────
    # In Bull regimes allow stocks recovering off recent lows (90% of SMA50)
    # In Neutral/Bear keep the tighter 95% filter to avoid weak stocks
    sma_floor = 0.90 if regime_code >= 3 else 0.95

    # ── FIX 2: regime-dependent volatility cap ─────────────────────────
    # Post-crisis recoveries have elevated vol — being too strict keeps
    # the screener empty for 2+ years after every crash
    # Bear regimes: tighter (don't want high-vol junk when defensive)
    # Bull regimes: slightly looser (momentum stocks can be volatile)
    vol_cap = {4: 60, 3: 70, 2: 75, 1: 65, 0: 55}.get(regime_code, 70)

    scores = []
    for ticker, price in universe.items():
        past = price[price.index <= as_of]
        # ── FIX 3: reduced minimum history 130 → 100 days ─────────────
        # 100 days is enough for a 50-day SMA with meaningful buffer
        # 130 was too strict and excluded stocks early in the backtest
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
            # Get real fundamental scores if available
            fund_quality = None
            fund_earnings = None
            if fund_snapshots is not None and not fund_snapshots.empty:
                try:
                    from build_fundamental_snapshots import get_fundamental_scores_at_date
                    fund_result = get_fundamental_scores_at_date(fund_snapshots, ticker, as_of)
                    if fund_result.get('has_fundamentals'):
                        fund_quality  = fund_result['quality_score']
                        fund_earnings = fund_result['earnings_score']
                except Exception:
                    pass

            # Blend real fundamentals with price proxies (70/30 when available)
            if fund_quality is not None:
                quality_final = fund_quality * 0.70 + max(0, 100 - vol) * 0.30
            else:
                quality_final = max(0, 100 - vol) * 0.5 + (50 if s(past.iloc[-1]) > s(past.rolling(50).mean().iloc[-1]) else 20) * 0.5

            if fund_earnings is not None:
                earnings_final = fund_earnings * 0.70 + earn * 0.30
            else:
                earnings_final = earn

            # Score-sensitive factor weights within Neutral
            if regime_code >= 3:   score = mom*0.35 + quality_final*0.15 + lowv*0.15 + earnings_final*0.25 + rs*10*0.10
            elif regime_code == 2: score = mom*0.30 + quality_final*0.20 + lowv*0.25 + earnings_final*0.15 + rs*10*0.10
            else:                  score = mom*0.10 + quality_final*0.30 + lowv*0.45 + earnings_final*0.10 + rs*10*0.05
            scores.append((ticker, score))
        except: continue

    scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scores[:n_pick]]


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


def run_walk_forward(nifty, vix, stock_data, n_months=72, fii_dii_df=None, fund_snapshots=None):
    print(f"\nRunning {n_months}-month walk-forward backtest...")
    try:    monthly = nifty.resample('ME').last().index
    except: monthly = nifty.resample('M').last().index
    monthly = monthly[-(n_months + 2):]

    results = []
    prev_tickers = None

    for i in range(len(monthly) - 1):
        from_date = monthly[i]
        to_date   = monthly[i + 1]
        regime    = compute_regime_score(nifty, vix, stock_data, from_date, fii_dii_df)
        code      = regime['code']
        score     = regime['score']
        eq_alloc  = {4:0.90, 3:0.80, 2:0.60, 1:0.15, 0:0.0}.get(code, 0.60)

        n_override = None
        if code == 3:  # Mild Bull momentum boost
            nc3 = nifty[nifty.index <= from_date]
            if len(nc3) > 63:
                m3b = (float(nc3.iloc[-1])/float(nc3.iloc[-63])-1)*100
                if m3b > 8.0:
                    eq_alloc = 0.88

        selected = select_stocks_at_date(stock_data, nifty, code, from_date, n_override, fund_snapshots, regime_score=score)
        port_ret = compute_monthly_return(selected, stock_data, from_date, to_date,
                                           prev_tickers, eq_alloc)

        nifty_p  = nifty[nifty.index <= from_date]
        nifty_f  = nifty[nifty.index > from_date]
        nifty_ret = 0.0
        if len(nifty_p) > 0 and len(nifty_f) > 0:
            tgt  = from_date + relativedelta(months=1)
            near = nifty_f[nifty_f.index <= tgt + timedelta(days=10)]
            if len(near) > 0:
                nifty_ret = s(near.iloc[-1]) / s(nifty_p.iloc[-1]) - 1

        results.append({
            'date':         str(from_date.date()),
            'regime_code':  code,
            'regime_score': score,
            'trend':        regime['trend'],
            'vola':         regime['vola'],
            'breadth':      regime['breadth'],
            'flow':         regime['flow'],
            'equity_alloc': eq_alloc,
            'n_stocks':     len(selected),
            'port_ret':     round(port_ret * 100, 3),
            'nifty_ret':    round(nifty_ret * 100, 3),
            'alpha':        round((port_ret - nifty_ret) * 100, 3),
            'tickers':      selected[:5],
        })
        prev_tickers = selected

        REGIME_LABELS = {4:'SBull', 3:'MBull', 2:'Neut', 1:'MBear', 0:'SBear'}
        print(f"  {str(from_date.date())} [{REGIME_LABELS[code]:5} {score:4.0f}] "
              f"stk={len(selected):2} eq={eq_alloc:.0%} | "
              f"Port={port_ret*100:+5.2f}% Nifty={nifty_ret*100:+5.2f}% "
              f"α={(port_ret-nifty_ret)*100:+5.2f}%")

    return pd.DataFrame(results)


def compute_metrics(df):
    df = df.copy()
    df['port_ret']  = pd.to_numeric(df['port_ret'], errors='coerce').fillna(0)
    df['nifty_ret'] = pd.to_numeric(df['nifty_ret'], errors='coerce').fillna(0)
    df['alpha']     = df['port_ret'] - df['nifty_ret']
    port_rets  = df['port_ret'].values / 100
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

    return {
        'period_months':     n,
        'port_total_ret':    round(port_cum*100, 2),
        'nifty_total_ret':   round(nifty_cum*100, 2),
        'port_ann_ret':      round(port_ann*100, 2),
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
    }


def print_report(metrics, df):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  20-YEAR BACKTEST RESULTS — 2005 to 2026")
    print(sep)
    print(f"  Period:           {metrics['period_months']} months ({metrics['period_months']//12} years)")
    print(f"  Strategy total:   {metrics['port_total_ret']:+.1f}%")
    print(f"  Nifty total:      {metrics['nifty_total_ret']:+.1f}%")
    print(f"  Alpha (total):    {metrics['port_total_ret']-metrics['nifty_total_ret']:+.1f}%")
    print(f"\n  Annualised:")
    print(f"  Strategy:         {metrics['port_ann_ret']:+.1f}% p.a.")
    print(f"  Nifty:            {metrics['nifty_ann_ret']:+.1f}% p.a.")
    print(f"  Alpha:            {metrics['alpha_ann']:+.1f}% p.a.")
    print(f"\n  Risk:")
    print(f"  Sharpe strategy:  {metrics['sharpe_strategy']:.2f}")
    print(f"  Sharpe Nifty:     {metrics['sharpe_benchmark']:.2f}")
    print(f"  Info ratio:       {metrics['information_ratio']:.2f}")
    print(f"  Max drawdown:     {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Nifty max dd:     {metrics['nifty_max_dd_pct']:.1f}%")
    print(f"  Calmar ratio:     {metrics['calmar_ratio']:.2f}")
    print(f"  Win rate:         {metrics['win_rate_vs_nifty']:.0f}%")
    print(f"  Beta:             {metrics['beta']:.2f}")
    print(f"\n  Regime breakdown:")
    for label, p in metrics['regime_breakdown'].items():
        print(f"  {label:<14} {p['months']:>3}M | "
              f"Port {p['port_avg']:+.2f}% | Nifty {p['nifty_avg']:+.2f}% | "
              f"α {p['alpha_avg']:+.2f}% | WR {p['win_rate']:.0f}%")
    print(f"\n  Period breakdown:")
    for label, p in metrics['period_breakdown'].items():
        print(f"  {label:<35} Port {p['port_total']:+5.1f}% Nifty {p['nifty_total']:+5.1f}% α {p['alpha_total']:+5.1f}%")
    print(sep)


def run():
    print("\n" + "="*65)
    print("  20-YEAR WALK-FORWARD BACKTEST")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*65)

    # Load fundamental snapshots for quality/earnings factors
    fund_snapshots = None
    fund_path = os.path.join(SCRIPT_DIR, 'fundamental_snapshots.csv')
    if os.path.exists(fund_path):
        try:
            fund_snapshots = pd.read_csv(fund_path, parse_dates=['snapshot_date'])
            n_stocks = fund_snapshots['ticker'].nunique()
            date_range = f"{fund_snapshots['snapshot_date'].min().date()} to {fund_snapshots['snapshot_date'].max().date()}"
            print(f"  Fundamentals: {n_stocks} stocks, {date_range}")
        except Exception as e:
            print(f"  Fundamental load failed: {e}")
    else:
        print("  Fundamentals: not found — using price proxies for quality/earnings")

    fii_dii_df = None
    fii_path   = os.path.join(SCRIPT_DIR, 'fii_dii_data.csv')
    if os.path.exists(fii_path):
        try:
            fii_dii_df = pd.read_csv(fii_path, parse_dates=['date'])
            fii_dii_df = fii_dii_df.sort_values('date').reset_index(drop=True)
            print(f"  FII/DII: {len(fii_dii_df)} days "
                  f"({fii_dii_df['date'].min().date()} to {fii_dii_df['date'].max().date()})")
            # Note: FII/DII data starts Jan 2022
            # Pre-2022 uses price-VIX proxy automatically (handled in compute_regime_score)
            print("  FII/DII: pre-2022 periods use price-VIX proxy (no data available)")
        except Exception as e:
            print(f"  FII/DII load failed: {e}")
    else:
        print("  FII/DII: not found — using price proxy for all periods")

    nifty, vix, stock_data = download_all_data(lookback_days=7500)
    df      = run_walk_forward(nifty, vix, stock_data, n_months=240, fii_dii_df=fii_dii_df, fund_snapshots=fund_snapshots)

    if df.empty:
        print("  No results generated")
        return

    metrics = compute_metrics(df)
    print_report(metrics, df)

    output = {
        'run_date': datetime.today().strftime('%Y-%m-%d'),
        'metrics':  metrics,
        'monthly':  df.to_dict('records'),
    }
    with open(BT_OUT, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {BT_OUT}")


if __name__ == "__main__":
    run()
