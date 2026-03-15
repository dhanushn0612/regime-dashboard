"""
Daily Regime Classifier Runner
================================
Auto-detects FII/DII data from fii_dii_scraper.py output.
If fii_dii_data.csv exists → uses real flow data.
If not → falls back to price-based proxy.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'public')
CURRENT_PATH = os.path.join(OUTPUT_DIR, 'regime_current.json')
HISTORY_PATH = os.path.join(OUTPUT_DIR, 'regime_history.json')
FII_DII_CSV  = os.path.join(os.path.dirname(__file__), 'fii_dii_data.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def s(x):
    if hasattr(x, 'item'):
        return float(x.item())
    return float(x)

def get_close(df):
    c = df['Close']
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.squeeze()


@dataclass
class RegimeSnapshot:
    date: str
    trend_score: float
    volatility_score: float
    breadth_score: float
    flow_score: float
    composite_score: float
    regime_label: str
    regime_code: int
    nifty_price: float
    india_vix: float
    recommended_action: str
    fii_dii_source: str
    dimension_signals: dict


REGIME_MAP = {
    (75, 101): (4, "Strong Bull",    "Full deployment — favor momentum + small/midcap"),
    (55,  75): (3, "Mild Bull",      "Moderate deployment — favor quality + large cap"),
    (40,  55): (2, "Neutral/Choppy", "Reduced exposure — tighten stops, avoid new entries"),
    (20,  40): (1, "Mild Bear",      "Defensive — rotate to debt/cash, reduce equity"),
    ( 0,  20): (0, "Strong Bear",    "Risk-Off — capital preservation mode"),
}

def score_to_regime(score):
    for (lo, hi), (code, label, action) in REGIME_MAP.items():
        if lo <= score < hi:
            return code, label, action
    return 0, "Strong Bear", "Risk-Off — capital preservation mode"


class TrendDimension:
    def score(self, nifty):
        close = get_close(nifty)
        signals = {}
        total = 0.0

        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        dist_50 = (s(close.iloc[-1]) - s(sma50.iloc[-1])) / s(sma50.iloc[-1]) * 100
        pts = 20 if dist_50 > 3 else 14 if dist_50 > 0 else 6 if dist_50 > -3 else 0
        total += pts
        signals['price_vs_sma50_pct'] = round(dist_50, 2)

        dist_200 = (s(close.iloc[-1]) - s(sma200.iloc[-1])) / s(sma200.iloc[-1]) * 100
        pts = 20 if dist_200 > 5 else 13 if dist_200 > 0 else 5 if dist_200 > -5 else 0
        total += pts
        signals['price_vs_sma200_pct'] = round(dist_200, 2)

        golden = s(sma50.iloc[-1]) > s(sma200.iloc[-1])
        pts = 15 if golden else 0
        total += pts
        signals['golden_cross'] = bool(golden)

        def roc(p):
            return (s(close.iloc[-1]) / s(close.iloc[-p]) - 1) * 100 if len(close) > p else 0.0

        for p, max_pts, thresh in [(21, 10, 3), (63, 10, 7), (126, 10, 12)]:
            r = roc(p)
            pts = max_pts if r > thresh else 6 if r > 0 else 2 if r > -thresh else 0
            total += pts
            signals[f'roc_{p}d_pct'] = round(r, 2)

        high_52w = s(close.rolling(252).max().iloc[-1])
        dist_52w = (s(close.iloc[-1]) / high_52w - 1) * 100
        pts = 15 if dist_52w > -5 else 10 if dist_52w > -10 else 4 if dist_52w > -20 else 0
        total += pts
        signals['dist_from_52w_high_pct'] = round(dist_52w, 2)

        return min(100, max(0, total)), signals


class VolatilityDimension:
    def score(self, nifty, vix):
        close = get_close(nifty)
        vix_close = get_close(vix).reindex(close.index, method='ffill').dropna()
        signals = {}
        total = 0.0

        current_vix = s(vix_close.iloc[-1])
        pts = 35 if current_vix < 13 else 28 if current_vix < 16 else 18 if current_vix < 20 else 10 if current_vix < 25 else 4 if current_vix < 30 else 0
        total += pts
        signals['india_vix'] = round(current_vix, 2)

        if len(vix_close) > 20:
            vix_chg = (s(vix_close.iloc[-1]) / s(vix_close.iloc[-20]) - 1) * 100
            pts = 20 if vix_chg < -15 else 15 if vix_chg < -5 else 10 if vix_chg < 5 else 4 if vix_chg < 15 else 0
            total += pts
            signals['vix_20d_change_pct'] = round(vix_chg, 2)

        returns = close.pct_change().dropna()
        rv_20d = s(returns.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
        rv_1y  = s(returns.rolling(252).std().iloc[-1]) * np.sqrt(252) * 100
        rv_ratio = rv_20d / rv_1y if rv_1y > 0 else 1.0
        pts = 25 if rv_ratio < 0.7 else 18 if rv_ratio < 0.9 else 12 if rv_ratio < 1.1 else 5 if rv_ratio < 1.4 else 0
        total += pts
        signals['rv_ratio'] = round(rv_ratio, 3)

        if len(vix_close) > 10:
            vov = s(vix_close.rolling(10).std().iloc[-1])
            pts = 20 if vov < 1.0 else 14 if vov < 2.0 else 7 if vov < 3.5 else 0
            total += pts
            signals['vix_stability_std'] = round(vov, 3)

        return min(100, max(0, total)), signals


class BreadthDimension:
    def score(self, components_data):
        signals = {}
        total = 0.0

        if not components_data:
            return 50.0, {'note': 'No breadth data'}

        closes = {}
        for t, df in components_data.items():
            if 'Close' in df.columns and len(df) > 200:
                closes[t] = get_close(df)

        above_50, above_200, advances = [], [], []
        for close in closes.values():
            try:
                curr   = s(close.iloc[-1])
                sma50  = s(close.rolling(50).mean().iloc[-1])
                sma200 = s(close.rolling(200).mean().iloc[-1])
                prev   = s(close.iloc[-2])
                above_50.append(curr > sma50)
                above_200.append(curr > sma200)
                advances.append(curr > prev)
            except:
                continue

        n = len(above_50)
        if n == 0:
            return 50.0, {'note': 'Could not compute breadth'}

        p50  = sum(above_50)  / n * 100
        p200 = sum(above_200) / n * 100
        adv  = sum(advances)  / n * 100

        pts = 35 if p50 > 70 else 25 if p50 > 55 else 15 if p50 > 40 else 7 if p50 > 30 else 0
        total += pts
        signals['pct_above_50dma'] = round(p50, 1)

        pts = 35 if p200 > 65 else 25 if p200 > 50 else 15 if p200 > 35 else 7 if p200 > 25 else 0
        total += pts
        signals['pct_above_200dma'] = round(p200, 1)

        pts = 30 if adv > 65 else 22 if adv > 55 else 14 if adv > 45 else 6 if adv > 35 else 0
        total += pts
        signals['advance_ratio_pct'] = round(adv, 1)
        signals['stocks_sampled'] = n

        return min(100, max(0, total)), signals


class FlowDimension:
    def score(self, nifty, vix, fii_dii_df=None):
        signals = {}
        total = 0.0
        close = get_close(nifty)
        vix_close = get_close(vix).reindex(close.index, method='ffill').dropna()

        # ── REAL FII/DII DATA ──────────────────────────────────────────
        if fii_dii_df is not None and len(fii_dii_df) >= 20:
            try:
                fii_20d = float(fii_dii_df['FII_Net'].tail(20).sum())
                dii_20d = float(fii_dii_df['DII_Net'].tail(20).sum())
                fii_5d  = float(fii_dii_df['FII_Net'].tail(5).sum())
                fii_10d = float(fii_dii_df['FII_Net'].tail(10).sum())

                # FII 20-day rolling net (30 pts)
                fii_cr = fii_20d / 100  # Already in Crores from NSE
                pts = 30 if fii_cr > 5000 else 22 if fii_cr > 1000 else 12 if fii_cr > -1000 else 4 if fii_cr > -5000 else 0
                total += pts
                signals['fii_20d_net_cr'] = round(fii_cr, 0)
                signals['fii_flow_pts'] = pts

                # FII vs DII positioning (20 pts)
                pts = 20 if fii_20d > 0 and dii_20d > 0 else \
                      14 if fii_20d > 0 and dii_20d < 0 else \
                       8 if fii_20d < 0 and dii_20d > 0 else 0
                total += pts
                signals['dii_20d_net_cr'] = round(dii_20d / 100, 0)
                signals['fii_dii_pts'] = pts

                # FII momentum — is recent flow accelerating? (15 pts)
                pts = 15 if fii_5d > fii_10d / 2 * 1.2 else \
                      10 if fii_5d > 0 else \
                       5 if fii_10d > 0 else 0
                total += pts
                signals['fii_trend_pts'] = pts
                signals['data_source'] = 'real_nse_fii_dii'

            except Exception as e:
                signals['fii_error'] = str(e)
                fii_dii_df = None

        # ── PRICE-BASED PROXY (fallback) ───────────────────────────────
        if fii_dii_df is None or len(fii_dii_df) < 20:
            r5d = (s(close.iloc[-1]) / s(close.iloc[-5]) - 1) * 100
            v5d = (s(vix_close.iloc[-1]) / s(vix_close.iloc[-5]) - 1) * 100
            pts = 65 if r5d > 1 and v5d < -5 else \
                  48 if r5d > 0 and v5d < 0 else \
                  30 if r5d > 0 else \
                  18 if v5d < 0 else 5
            total += pts
            signals['data_source'] = 'price_proxy'
            signals['note'] = 'Run fii_dii_scraper.py to unlock real flow data'

        # ── VIX SLOPE SENTIMENT (35 pts — always active) ──────────────
        if len(vix_close) > 20:
            slope = (s(vix_close.iloc[-1]) - s(vix_close.iloc[-20])) / 20
            pts = 35 if slope < -0.2 else 25 if slope < 0 else 15 if slope < 0.2 else 6 if slope < 0.5 else 0
            total += pts
            signals['vix_slope_20d'] = round(slope, 4)

        return min(100, max(0, total)), signals


NIFTY500_SAMPLE = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","HCLTECH.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","BAJFINANCE.NS","NESTLEIND.NS","ULTRACEMCO.NS",
    "PIDILITIND.NS","HAVELLS.NS","VOLTAS.NS","MUTHOOTFIN.NS","COFORGE.NS",
    "PERSISTENT.NS","TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS",
    "ONGC.NS","NTPC.NS","POWERGRID.NS","WIPRO.NS","TECHM.NS",
    "DRREDDY.NS","CIPLA.NS","FEDERALBNK.NS","INDHOTEL.NS","ADANIPORTS.NS",
]


def load_fii_dii():
    """Load FII/DII CSV if it exists."""
    if not os.path.exists(FII_DII_CSV):
        return None
    try:
        df = pd.read_csv(FII_DII_CSV, parse_dates=['date'])
        if len(df) >= 20:
            print(f"  ✓ FII/DII data loaded: {len(df)} days (up to {df['date'].max().strftime('%Y-%m-%d')})")
            return df
    except Exception as e:
        print(f"  ✗ FII/DII load error: {e}")
    return None


def run():
    print(f"\n{'='*55}")
    print(f"  REGIME CLASSIFIER — {datetime.today().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*55}\n")

    end   = datetime.today()
    start = end - timedelta(days=600)

    print("Downloading Nifty 50...")
    nifty = flatten(yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True))
    print(f"  ✓ {len(nifty)} days")

    print("Downloading India VIX...")
    vix = flatten(yf.download("^INDIAVIX", start=start, end=end, progress=False, auto_adjust=True))
    print(f"  ✓ {len(vix)} days")

    print(f"Downloading {len(NIFTY500_SAMPLE)} breadth stocks...")
    components = {}
    for t in NIFTY500_SAMPLE:
        try:
            df = flatten(yf.download(t, start=start, end=end, progress=False, auto_adjust=True))
            if len(df) > 100:
                components[t] = df
        except:
            pass
    print(f"  ✓ {len(components)} stocks loaded")

    # Auto-load FII/DII if available
    print("Loading FII/DII data...")
    fii_dii_df = load_fii_dii()
    if fii_dii_df is None:
        print("  ⚠ No FII/DII data — using price proxy (run fii_dii_scraper.py first)")

    fii_dii_source = "real_nse_fii_dii" if fii_dii_df is not None else "price_proxy"

    # Score dimensions
    ts, td = TrendDimension().score(nifty)
    vs, vd = VolatilityDimension().score(nifty, vix)
    bs, bd = BreadthDimension().score(components)
    fs, fd = FlowDimension().score(nifty, vix, fii_dii_df)

    composite = ts * 0.30 + vs * 0.25 + bs * 0.25 + fs * 0.20
    code, label, action = score_to_regime(composite)

    current_price = s(get_close(nifty).iloc[-1])
    current_vix   = s(get_close(vix).iloc[-1]) if len(vix) > 0 else 0.0

    snap = RegimeSnapshot(
        date               = str(nifty.index[-1].date()),
        trend_score        = round(ts, 1),
        volatility_score   = round(vs, 1),
        breadth_score      = round(bs, 1),
        flow_score         = round(fs, 1),
        composite_score    = round(composite, 1),
        regime_label       = label,
        regime_code        = code,
        nifty_price        = round(current_price, 2),
        india_vix          = round(current_vix, 2),
        recommended_action = action,
        fii_dii_source     = fii_dii_source,
        dimension_signals  = {'trend': td, 'volatility': vd, 'breadth': bd, 'flow': fd},
    )

    print("\nBuilding 2-year history...")
    hist_records = []
    weekly_dates = nifty['2023-01-01':].resample('W').last().index

    for d in weekly_dates:
        try:
            n_sl = nifty[nifty.index <= d]
            v_sl = vix[vix.index <= d]
            if len(n_sl) < 200:
                continue

            # Slice FII/DII to date
            f_sl = fii_dii_df[fii_dii_df['date'] <= d] if fii_dii_df is not None else None

            t2, _ = TrendDimension().score(n_sl)
            v2, _ = VolatilityDimension().score(n_sl, v_sl)
            b2, _ = BreadthDimension().score(components)
            f2, _ = FlowDimension().score(n_sl, v_sl, f_sl)
            c2    = t2 * 0.30 + v2 * 0.25 + b2 * 0.25 + f2 * 0.20
            rc, rl, _ = score_to_regime(c2)

            hist_records.append({
                'date':             str(d.date()),
                'nifty_price':      round(s(get_close(n_sl).iloc[-1]), 2),
                'india_vix':        round(s(get_close(v_sl).iloc[-1]) if len(v_sl) > 0 else 0, 2),
                'trend_score':      round(t2, 1),
                'volatility_score': round(v2, 1),
                'breadth_score':    round(b2, 1),
                'flow_score':       round(f2, 1),
                'composite_score':  round(c2, 1),
                'regime_label':     rl,
                'regime_code':      rc,
            })
        except:
            continue

    print(f"  ✓ {len(hist_records)} weekly snapshots")

    with open(CURRENT_PATH, 'w') as f:
        json.dump(asdict(snap), f, indent=2, default=str)

    with open(HISTORY_PATH, 'w') as f:
        json.dump(hist_records, f, default=str)

    print(f"\n✓ Saved: {CURRENT_PATH}")
    print(f"✓ Saved: {HISTORY_PATH}")
    print(f"\n  Regime:       {label} ({composite:.1f}/100)")
    print(f"  Nifty:        {current_price:,.0f}  |  VIX: {current_vix:.1f}")
    print(f"  Flow Source:  {fii_dii_source}")
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    run()
