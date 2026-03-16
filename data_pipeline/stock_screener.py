"""
Stock Screener — Layer 3
==========================
Multi-factor stock screener for Nifty 500 universe.
Hybrid: rule-based quality filter → Random Forest ML ranking.

Factors:
  Momentum    — price momentum across 1M/3M/6M/12M
  Quality     — ROE proxy, debt ratio, profitability
  Low Vol     — realised volatility, downside deviation
  Earnings    — revenue/earnings growth proxy from price

Architecture:
  Step 1 — Load Nifty 500 universe (price data via yfinance)
  Step 2 — Rule-based quality filter (remove junk)
  Step 3 — Score all four factors per stock
  Step 4 — Random Forest ranks stocks by predicted forward return
  Step 5 — Regime-dependent output (top N stocks)
  Step 6 — Save to public/screener_current.json

Usage:
    python data_pipeline/stock_screener.py

Output:
    public/screener_current.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

# Fundamental data integration
try:
    from fundamental_data import (
        fetch_fundamentals_batch,
        compute_real_quality_score,
        compute_real_earnings_score,
        blend_scores,
    )
    HAS_FUNDAMENTALS = True
except ImportError:
    HAS_FUNDAMENTALS = False

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, '..', 'public')
SCREENER_OUT = os.path.join(OUTPUT_DIR, 'screener_current.json')
REGIME_CURR  = os.path.join(OUTPUT_DIR, 'regime_current.json')
SECTOR_CURR  = os.path.join(OUTPUT_DIR, 'sector_current.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── REGIME-DEPENDENT STOCK COUNT ──────────────────────────────────────
REGIME_STOCK_COUNT = {
    4: 15,   # Strong Bull  — top 15 stocks
    3: 10,   # Mild Bull    — top 10 stocks
    2: 5,    # Neutral      — top 5 (high conviction only)
    1: 0,    # Mild Bear    — watchlist only, no deployment
    0: 0,    # Strong Bear  — no stocks
}

# ── NIFTY 500 SAMPLE UNIVERSE (100 liquid stocks) ─────────────────────
# Representative cross-section of Nifty 500
NIFTY500_UNIVERSE = [
    # Large cap — Financials
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "FEDERALBNK.NS", "INDUSINDBK.NS", "BANDHANBNK.NS",
    # Large cap — IT
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "OFSS.NS",
    # Large cap — Consumer
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
    "GODREJCP.NS", "MARICO.NS", "COLPAL.NS", "EMAMILTD.NS", "TATACONSUM.NS",
    # Large cap — Auto
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "TVSMOTORS.NS", "ASHOKLEY.NS", "BOSCHLTD.NS", "MOTHERSON.NS",
    # Large cap — Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS",
    "TORNTPHARM.NS", "ALKEM.NS", "LUPIN.NS", "IPCALAB.NS", "GLENMARK.NS",
    # Large cap — Energy & Metals
    "RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "VEDL.NS", "SAIL.NS", "NMDC.NS", "ADANIPORTS.NS",
    # Large cap — Infra & Capital Goods
    "LT.NS", "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS",
    "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "CUMMINSIND.NS", "BHEL.NS",
    # Large cap — Diversified
    "TITAN.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "BERGEPAINT.NS", "WHIRLPOOL.NS",
    "VOLTAS.NS", "CROMPTON.NS", "DIXON.NS", "AMBER.NS", "POLYCAB.NS",
    # Mid cap — High quality
    "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS", "KPITTECH.NS", "TATAELXSI.NS",
    "MUTHOOTFIN.NS", "CHOLAFIN.NS", "MANAPPURAM.NS", "LICHSGFIN.NS", "PEL.NS",
    # Mid cap — Consumer & Retail
    "DMART.NS", "TRENT.NS", "NYKAA.NS", "ZOMATO.NS", "PAYTM.NS",
    "INDHOTEL.NS", "LEMONTREE.NS", "CHALET.NS", "JUBLFOOD.NS", "DEVYANI.NS",
]

# Remove duplicates
NIFTY500_UNIVERSE = list(dict.fromkeys(NIFTY500_UNIVERSE))


# ── HELPERS ───────────────────────────────────────────────────────────
def s(x):
    if hasattr(x, 'item'):
        return float(x.item())
    return float(x)

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_close(df):
    c = df['Close']
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.squeeze()

def get_volume(df):
    v = df['Volume']
    if isinstance(v, pd.DataFrame):
        v = v.iloc[:, 0]
    return v.squeeze()


# ── STEP 1: DOWNLOAD STOCK DATA ───────────────────────────────────────
def download_universe(tickers: list, lookback_days: int = 400) -> dict:
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    print(f"Downloading {len(tickers)} stocks...")
    data = {}
    failed = 0

    for ticker in tickers:
        try:
            df = flatten(yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=True
            ))
            if len(df) > 150:
                data[ticker] = df
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"  Loaded: {len(data)}/{len(tickers)}  Failed: {failed}")
    return data


# ── STEP 2: RULE-BASED QUALITY FILTER ─────────────────────────────────
def quality_filter(stock_data: dict, regime_code: int) -> tuple:
    """
    Remove stocks that fail basic quality rules.
    Rules:
      1. Minimum liquidity — avg daily volume > 500k shares
      2. No extreme volatility — annualised vol < 80%
      3. Not in severe downtrend — price not >40% below 52w high
      4. Positive 6M return (avoid value traps in bear markets)
    """
    passing  = []
    excluded = {}

    for ticker, df in stock_data.items():
        close  = get_close(df)
        vol    = get_volume(df)
        rets   = close.pct_change().dropna()
        reasons = []

        # Rule 1: Liquidity
        avg_vol = s(vol.tail(20).mean())
        if avg_vol < 500_000:
            reasons.append(f"low liquidity ({avg_vol/1e6:.1f}M avg vol)")

        # Rule 2: Volatility cap
        ann_vol = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
        if ann_vol > 80:
            reasons.append(f"excessive vol ({ann_vol:.0f}%)")

        # Rule 3: Not in severe downtrend
        high_52w = s(close.rolling(252).max().iloc[-1])
        drawdown = (s(close.iloc[-1]) / high_52w - 1) * 100
        if drawdown < -40:
            reasons.append(f"severe drawdown ({drawdown:.0f}%)")

        # Rule 4: In bear/neutral regime, require positive 6M return
        if regime_code <= 2 and len(close) > 126:
            ret_6m = (s(close.iloc[-1]) / s(close.iloc[-126]) - 1) * 100
            if ret_6m < -15:
                reasons.append(f"6M return {ret_6m:.0f}%")

        if reasons:
            excluded[ticker] = reasons
        else:
            passing.append(ticker)

    print(f"  Quality filter: {len(passing)} pass, {len(excluded)} excluded")
    return passing, excluded


# ── STEP 3: FACTOR SCORING ────────────────────────────────────────────
def compute_factors(stock_data: dict, tickers: list,
                    nifty_close: pd.Series,
                    fundamentals: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute all four factor scores per stock.

    Factor 1 — Momentum (0-100):
        Composite of 1M/3M/6M/12M returns, weighted toward medium-term
        Cross-sectionally ranked and normalised

    Factor 2 — Quality (0-100):
        Proxy from price stability, consistency of trend
        True quality (ROE, D/E) requires fundamental data
        We use: earnings consistency proxy + price/SMA relationship

    Factor 3 — Low Volatility (0-100):
        Inverted annualised vol, downside deviation, max drawdown
        Lower vol = higher score

    Factor 4 — Earnings Growth (0-100):
        Revenue/earnings growth proxy from price acceleration
        Uses price momentum vs sector benchmark
    """
    rows = []

    for ticker in tickers:
        if ticker not in stock_data:
            continue
        df    = stock_data[ticker]
        close = get_close(df)
        vol   = get_volume(df)
        rets  = close.pct_change().dropna()

        def roc(p):
            return (s(close.iloc[-1]) / s(close.iloc[-p]) - 1) * 100 if len(close) > p else 0.0

        # ── MOMENTUM FACTOR ───────────────────────────────────────────
        ret_1m  = roc(21)
        ret_3m  = roc(63)
        ret_6m  = roc(126)
        ret_12m = roc(252)

        # Weighted momentum composite (medium-term biased)
        mom_composite = (
            ret_1m  * 0.15 +
            ret_3m  * 0.35 +
            ret_6m  * 0.30 +
            ret_12m * 0.20
        )

        # Trend consistency — how many of 4 lookbacks are positive
        trend_consistency = sum([
            ret_1m > 0, ret_3m > 0, ret_6m > 0, ret_12m > 0
        ]) / 4

        # Relative strength vs Nifty
        nifty_3m = (s(nifty_close.iloc[-1]) / s(nifty_close.iloc[-63]) - 1) * 100 if len(nifty_close) > 63 else 0
        rs_3m    = ret_3m - nifty_3m

        # ── QUALITY FACTOR (price-based proxy) ────────────────────────
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        above_50dma  = float(s(close.iloc[-1]) > s(sma50.iloc[-1]))
        above_200dma = float(s(close.iloc[-1]) > s(sma200.iloc[-1]))
        golden_cross = float(s(sma50.iloc[-1]) > s(sma200.iloc[-1]))

        # Price acceleration: is momentum improving?
        ret_1m_prev = (s(close.iloc[-21]) / s(close.iloc[-42]) - 1) * 100 if len(close) > 42 else 0
        accel = ret_1m - ret_1m_prev  # Positive = accelerating

        # Volume trend: rising price + rising volume = quality move
        vol_20  = s(vol.tail(20).mean())
        vol_60  = s(vol.tail(60).mean())
        vol_trend = float(vol_20 > vol_60)  # Volume expanding

        # Price-based proxy (always computed)
        quality_price = (
            above_50dma  * 25 +
            above_200dma * 25 +
            golden_cross * 20 +
            (min(max(accel, -5), 5) + 5) / 10 * 15 +
            vol_trend * 15
        )

        # Real fundamental quality (ROE, D/E, operating margin)
        quality_real = None
        if fundamentals is not None and not fundamentals.empty and HAS_FUNDAMENTALS:
            real_q = compute_real_quality_score(fundamentals, ticker)
            quality_real = real_q.get('quality_real')

        # Blend: 70% real, 30% price proxy
        quality_composite = blend_scores(quality_price, quality_real, real_weight=0.7)

        # ── LOW VOLATILITY FACTOR ──────────────────────────────────────
        ann_vol_20 = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
        ann_vol_60 = s(rets.rolling(60).std().iloc[-1]) * np.sqrt(252) * 100

        # Downside deviation (semi-deviation)
        neg_rets      = rets[rets < 0]
        downside_dev  = float(neg_rets.std() * np.sqrt(252) * 100) if len(neg_rets) > 5 else ann_vol_20

        # Max drawdown (rolling 1Y)
        roll_max = close.rolling(252).max()
        drawdown = ((close - roll_max) / roll_max * 100)
        max_dd   = abs(s(drawdown.min()))

        # Low vol score: invert and normalise (lower vol = higher score)
        vol_score = max(0, 100 - ann_vol_20)      # 0% vol = 100, 100% vol = 0
        dd_score  = max(0, 100 - max_dd * 2)       # 0% dd = 100, 50% dd = 0
        lowvol_composite = vol_score * 0.5 + dd_score * 0.5

        # ── EARNINGS GROWTH FACTOR (price acceleration proxy) ─────────
        # True earnings growth needs quarterly data from screeners
        # Proxy: sustained price momentum + volume confirmation
        # Stocks that "earn" their price moves show consistent momentum

        # Price earnings proxy: 12M return divided into 4 quarters
        q1 = roc(63)   # Last 3M
        q2 = (s(close.iloc[-63]) / s(close.iloc[-126]) - 1) * 100 if len(close) > 126 else 0
        q3 = (s(close.iloc[-126]) / s(close.iloc[-189]) - 1) * 100 if len(close) > 189 else 0
        q4 = (s(close.iloc[-189]) / s(close.iloc[-252]) - 1) * 100 if len(close) > 252 else 0

        # Growth consistency: how many quarters are positive
        growth_consistency = sum([q1 > 0, q2 > 0, q3 > 0, q4 > 0]) / 4

        # Trend of growth: is recent quarter better than previous?
        growth_trend = float(q1 > q2)

        # 52-week high proximity (growing stocks make new highs)
        high_52w     = s(close.rolling(252).max().iloc[-1])
        high_prox    = (s(close.iloc[-1]) / high_52w) * 100  # 100 = at 52w high

        # Price-based proxy
        earnings_price = (
            growth_consistency * 40 +
            growth_trend * 20 +
            (high_prox - 60) * 1.0
        )
        earnings_price = max(0, min(100, earnings_price))

        # Real fundamental earnings (revenue growth, net margin, FCF)
        earnings_real = None
        if fundamentals is not None and not fundamentals.empty and HAS_FUNDAMENTALS:
            real_e = compute_real_earnings_score(fundamentals, ticker)
            earnings_real = real_e.get('earnings_real')

        # Blend: 70% real, 30% price proxy
        earnings_composite = blend_scores(earnings_price, earnings_real, real_weight=0.7)
        earnings_composite = max(0, min(100, earnings_composite))

        rows.append({
            'ticker':             ticker,
            'name':               ticker.replace('.NS', ''),

            # Raw inputs (for display)
            'ret_1m':             round(ret_1m, 2),
            'ret_3m':             round(ret_3m, 2),
            'ret_6m':             round(ret_6m, 2),
            'ret_12m':            round(ret_12m, 2),
            'rs_vs_nifty_3m':     round(rs_3m, 2),
            'ann_vol_pct':        round(ann_vol_20, 1),
            'max_drawdown_pct':   round(max_dd, 1),
            'dist_52w_high_pct':  round((s(close.iloc[-1]) / high_52w - 1) * 100, 1),

            # Factor scores (for ML)
            'f_momentum':         round(mom_composite, 2),
            'f_quality':          round(quality_composite, 2),
            'f_lowvol':           round(lowvol_composite, 2),
            'f_earnings':         round(earnings_composite, 2),

            # Helper features
            'trend_consistency':  round(trend_consistency, 2),
            'above_50dma':        above_50dma,
            'above_200dma':       above_200dma,
            'golden_cross':       golden_cross,
            'vol_trend':          vol_trend,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Cross-sectional normalisation: rank each factor 0-100
    for col in ['f_momentum', 'f_quality', 'f_lowvol', 'f_earnings']:
        df[f'{col}_rank'] = df[col].rank(pct=True) * 100

    return df


# ── STEP 4: RANDOM FOREST RANKING ─────────────────────────────────────
def rank_with_random_forest(factors_df: pd.DataFrame,
                             regime_code: int) -> pd.DataFrame:
    """
    Use Random Forest to rank stocks.

    In regimes where we have enough data, we train a simple RF on
    cross-sectional factor scores → predict composite score.

    In practice with only current data (no labels), we use the RF
    as a weighted factor combiner where weights are regime-adjusted.

    Full supervised training (with forward returns as labels) requires
    historical factor data — this is added in Layer 4.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        HAS_RF = True
    except ImportError:
        HAS_RF = False

    FEATURE_COLS = [
        'f_momentum_rank', 'f_quality_rank',
        'f_lowvol_rank', 'f_earnings_rank',
        'trend_consistency', 'above_50dma',
        'above_200dma', 'golden_cross', 'vol_trend',
        'rs_vs_nifty_3m',
    ]

    # Regime-adjusted factor weights
    # Bull: momentum + earnings dominate
    # Bear: quality + low-vol dominate
    REGIME_WEIGHTS = {
        4: {'f_momentum_rank': 0.35, 'f_quality_rank': 0.20, 'f_lowvol_rank': 0.15, 'f_earnings_rank': 0.30},
        3: {'f_momentum_rank': 0.30, 'f_quality_rank': 0.25, 'f_lowvol_rank': 0.20, 'f_earnings_rank': 0.25},
        2: {'f_momentum_rank': 0.20, 'f_quality_rank': 0.30, 'f_lowvol_rank': 0.30, 'f_earnings_rank': 0.20},
        1: {'f_momentum_rank': 0.10, 'f_quality_rank': 0.35, 'f_lowvol_rank': 0.45, 'f_earnings_rank': 0.10},
        0: {'f_momentum_rank': 0.10, 'f_quality_rank': 0.35, 'f_lowvol_rank': 0.45, 'f_earnings_rank': 0.10},
    }

    weights = REGIME_WEIGHTS.get(regime_code, REGIME_WEIGHTS[2])

    if HAS_RF and len(factors_df) >= 20:
        # Use RF as an unsupervised ranker via synthetic labels
        # Synthetic label = weighted factor composite (RF learns non-linear interactions)
        factors_df['synthetic_score'] = sum(
            factors_df[col] * w for col, w in weights.items()
        )

        X = factors_df[FEATURE_COLS].fillna(0)
        y = factors_df['synthetic_score']

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)
        factors_df['ml_score'] = rf.predict(X)
        factors_df['model_used'] = 'random_forest'

        # Feature importance
        importance = dict(zip(FEATURE_COLS, rf.feature_importances_))
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        feat_strs = [k.replace("f_","").replace("_rank","") + "(" + str(round(v,2)) + ")" for k,v in top]
        print("  RF top features: " + ", ".join(feat_strs))

    else:
        # Fallback: pure weighted composite
        factors_df['ml_score'] = sum(
            factors_df[col] * w for col, w in weights.items()
        )
        factors_df['model_used'] = 'weighted_composite'
        print(f"  Using weighted composite (RF not available or insufficient data)")

    return factors_df.sort_values('ml_score', ascending=False).reset_index(drop=True)


# ── STEP 5: BUILD OUTPUT ──────────────────────────────────────────────
def build_screener_output(ranked_df: pd.DataFrame,
                           excluded: dict,
                           regime_code: int,
                           regime_label: str,
                           composite_score: float,
                           active_sectors: list) -> dict:

    n_stocks = REGIME_STOCK_COUNT.get(regime_code, 0)

    selected = ranked_df.head(n_stocks) if n_stocks > 0 else pd.DataFrame()

    stocks = []
    if not selected.empty:
        for _, row in selected.iterrows():
            stocks.append({
                'ticker':          row['ticker'],
                'name':            row['name'],
                'ml_score':        round(float(row['ml_score']), 2),
                'f_momentum':      round(float(row['f_momentum_rank']), 1),
                'f_quality':       round(float(row['f_quality_rank']), 1),
                'f_lowvol':        round(float(row['f_lowvol_rank']), 1),
                'f_earnings':      round(float(row['f_earnings_rank']), 1),
                'ret_1m':          round(float(row['ret_1m']), 2),
                'ret_3m':          round(float(row['ret_3m']), 2),
                'ret_6m':          round(float(row['ret_6m']), 2),
                'rs_vs_nifty_3m':  round(float(row['rs_vs_nifty_3m']), 2),
                'ann_vol_pct':     round(float(row['ann_vol_pct']), 1),
                'max_dd_pct':      round(float(row['max_drawdown_pct']), 1),
                'dist_52w_high':   round(float(row['dist_52w_high_pct']), 1),
                'model_used':      row['model_used'],
            })

    return {
        'date':              datetime.today().strftime('%Y-%m-%d'),
        'regime_code':       regime_code,
        'regime_label':      regime_label,
        'composite_score':   composite_score,
        'stocks_screened':   len(ranked_df) + len(excluded),
        'stocks_passed':     len(ranked_df),
        'stocks_selected':   n_stocks,
        'active_sectors':    active_sectors,
        'stocks':            stocks,
        'excluded_count':    len(excluded),
        'status':            'active' if n_stocks > 0 else 'watchlist_only',
        'note':              (
            f"Top {n_stocks} stocks selected for deployment"
            if n_stocks > 0
            else "Regime too weak for deployment — watchlist mode only"
        ),
    }


# ── PRINT REPORT ──────────────────────────────────────────────────────
def print_report(output: dict):
    sep = "=" * 60
    print("")
    print(sep)
    print("  STOCK SCREENER -- " + output['date'])
    print(sep)
    print("  Regime:   " + output['regime_label'] + " (" + str(output['composite_score']) + "/100)")
    print("  Universe: " + str(output['stocks_screened']) + " stocks")
    print("  Passed:   " + str(output['stocks_passed']) + " stocks")
    print("  Selected: " + str(output['stocks_selected']) + " stocks")
    print("  Status:   " + output['status'].upper())
    if output['stocks']:
        print("")
        print("  TOP STOCKS:")
        print("  {:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            "TICKER", "SCORE", "MOM", "QUAL", "VOL", "EARN", "3M RS"))
        print("  " + "-" * 58)
        for s in output['stocks'][:10]:
            print("  {:<12} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f} {:>7.1f}%".format(
                s['name'], s['ml_score'],
                s['f_momentum'], s['f_quality'],
                s['f_lowvol'], s['f_earnings'],
                s['rs_vs_nifty_3m']
            ))
    else:
        print("")
        print("  No stocks selected -- " + output['note'])
    print(sep)
    print("")


# ── MAIN ──────────────────────────────────────────────────────────────
def run():
    print("")
    print("=" * 60)
    print("  STOCK SCREENER -- " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("=" * 60)
    print("")

    # Load regime
    regime_code     = 2
    regime_label    = "Neutral/Choppy"
    composite_score = 50.0

    if os.path.exists(REGIME_CURR):
        with open(REGIME_CURR) as f:
            reg = json.load(f)
        regime_code     = reg.get('regime_code', 2)
        regime_label    = reg.get('regime_label', 'Neutral/Choppy')
        composite_score = reg.get('composite_score', 50.0)
        print("  Regime: " + regime_label + " (" + str(composite_score) + "/100)")

    # Load active sectors from Layer 2
    active_sectors = []
    if os.path.exists(SECTOR_CURR):
        with open(SECTOR_CURR) as f:
            sec = json.load(f)
        active_sectors = [a['sector'] for a in sec.get('allocations', [])]
        if active_sectors:
            print("  Active sectors: " + ", ".join(active_sectors))
        else:
            print("  Active sectors: None (cash mode)")

    # Step 1: Download universe
    print("")
    stock_data = download_universe(NIFTY500_UNIVERSE, lookback_days=400)

    if not stock_data:
        print("  ERROR: No stock data downloaded")
        return

    # Step 2: Quality filter
    print("")
    print("Applying quality filter...")
    nifty = flatten(yf.download("^NSEI", start=datetime.today()-timedelta(days=400),
                                 end=datetime.today(), progress=False, auto_adjust=True))
    nifty_close = nifty['Close'].squeeze()
    if isinstance(nifty_close, pd.DataFrame):
        nifty_close = nifty_close.iloc[:, 0]

    passing, excluded = quality_filter(stock_data, regime_code)

    # Step 3: Fetch fundamentals
    fundamentals_df = None
    if HAS_FUNDAMENTALS:
        print("")
        print("Loading fundamental data...")
        try:
            fundamentals_df = fetch_fundamentals_batch(
                [t for t in passing if t in stock_data],
                force_refresh=False
            )
            if fundamentals_df is not None and not fundamentals_df.empty:
                print("  Fundamentals: " + str(len(fundamentals_df)) + " stocks")
            else:
                print("  No fundamental data — using price proxies")
                fundamentals_df = None
        except Exception as e:
            print("  Fundamental fetch error: " + str(e))
            fundamentals_df = None

    # Step 3b: Factor scoring
    print("")
    print("Computing factor scores...")
    passing_data = {t: stock_data[t] for t in passing if t in stock_data}
    factors_df   = compute_factors(passing_data, passing, nifty_close, fundamentals_df)

    if factors_df.empty:
        print("  No stocks survived factor scoring")
        output = build_screener_output(
            pd.DataFrame(), excluded, regime_code,
            regime_label, composite_score, active_sectors
        )
        with open(SCREENER_OUT, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        return

    print("  Scored " + str(len(factors_df)) + " stocks")

    # Step 4: Random Forest ranking
    print("")
    print("Ranking with Random Forest...")
    ranked_df = rank_with_random_forest(factors_df, regime_code)

    # Step 5: Build output
    output = build_screener_output(
        ranked_df, excluded, regime_code,
        regime_label, composite_score, active_sectors
    )

    # Print report
    print_report(output)

    # Save
    with open(SCREENER_OUT, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("Saved: " + SCREENER_OUT)
    print("")


if __name__ == "__main__":
    run()
