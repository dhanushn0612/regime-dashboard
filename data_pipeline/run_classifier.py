"""
Daily Regime Classifier Runner
================================
This script is called by GitHub Actions every weekday morning.
It downloads fresh market data, runs the classifier, and writes
two JSON files into /public — which Vercel serves to the dashboard.

NO manual intervention needed after initial setup.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

warnings.filterwarnings('ignore')

# ── OUTPUT PATHS (relative to repo root) ─────────────────────────────
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), '..', 'public')
CURRENT_PATH   = os.path.join(OUTPUT_DIR, 'regime_current.json')
HISTORY_PATH   = os.path.join(OUTPUT_DIR, 'regime_history.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── PASTE YOUR FULL regime_classifier.py CODE BELOW ──────────────────
# (Copy everything from your regime_classifier.py EXCEPT the if __name__ == "__main__" block)
# For convenience, the key classes are re-imported here inline.

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
        close = nifty['Close']
        signals = {}
        total = 0.0

        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        dist_50 = (close.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1] * 100
        pts = 20 if dist_50 > 3 else 14 if dist_50 > 0 else 6 if dist_50 > -3 else 0
        total += pts
        signals['price_vs_sma50_pct'] = round(float(dist_50), 2)

        dist_200 = (close.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1] * 100
        pts = 20 if dist_200 > 5 else 13 if dist_200 > 0 else 5 if dist_200 > -5 else 0
        total += pts
        signals['price_vs_sma200_pct'] = round(float(dist_200), 2)

        golden = bool(sma50.iloc[-1] > sma200.iloc[-1])
        pts = 15 if golden else 0
        total += pts
        signals['golden_cross'] = golden

        def roc(p):
            return float((close.iloc[-1] / close.iloc[-p] - 1) * 100) if len(close) > p else 0.0

        for p, max_pts, thresh in [(21,10,3), (63,10,7), (126,10,12)]:
            r = roc(p)
            pts = max_pts if r > thresh else 6 if r > 0 else 2 if r > -thresh else 0
            total += pts
            signals[f'roc_{p}d_pct'] = round(r, 2)

        high_52w = close.rolling(252).max().iloc[-1]
        dist_52w = float((close.iloc[-1] / high_52w - 1) * 100)
        pts = 15 if dist_52w > -5 else 10 if dist_52w > -10 else 4 if dist_52w > -20 else 0
        total += pts
        signals['dist_from_52w_high_pct'] = round(dist_52w, 2)

        return min(100, max(0, total)), signals


class VolatilityDimension:
    def score(self, nifty, vix):
        close = nifty['Close']
        signals = {}
        total = 0.0

        vix_close = vix['Close'] if 'Close' in vix.columns else vix.iloc[:, 0]
        vix_close = vix_close.reindex(close.index, method='ffill').dropna()

        current_vix = float(vix_close.iloc[-1])
        pts = 35 if current_vix < 13 else 28 if current_vix < 16 else 18 if current_vix < 20 else 10 if current_vix < 25 else 4 if current_vix < 30 else 0
        total += pts
        signals['india_vix'] = round(current_vix, 2)

        if len(vix_close) > 20:
            vix_chg = float((vix_close.iloc[-1] / vix_close.iloc[-20] - 1) * 100)
            pts = 20 if vix_chg < -15 else 15 if vix_chg < -5 else 10 if vix_chg < 5 else 4 if vix_chg < 15 else 0
            total += pts
            signals['vix_20d_change_pct'] = round(vix_chg, 2)

        returns = close.pct_change().dropna()
        rv_20d = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
        rv_1y  = float(returns.rolling(252).std().iloc[-1] * np.sqrt(252) * 100)
        rv_ratio = rv_20d / rv_1y if rv_1y > 0 else 1.0
        pts = 25 if rv_ratio < 0.7 else 18 if rv_ratio < 0.9 else 12 if rv_ratio < 1.1 else 5 if rv_ratio < 1.4 else 0
        total += pts
        signals['rv_ratio'] = round(rv_ratio, 3)

        if len(vix_close) > 10:
            vov = float(vix_close.rolling(10).std().iloc[-1])
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

        closes = {t: df['Close'] for t, df in components_data.items()
                  if 'Close' in df.columns and len(df) > 200}

        above_50, above_200, advances = [], [], []
        for close in closes.values():
            try:
                curr = close.iloc[-1]
                above_50.append(curr > close.rolling(50).mean().iloc[-1])
                above_200.append(curr > close.rolling(200).mean().iloc[-1])
                advances.append(curr > close.iloc[-2])
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
        close = nifty['Close']
        vix_close = vix['Close'] if 'Close' in vix.columns else vix.iloc[:, 0]
        vix_close = vix_close.reindex(close.index, method='ffill').dropna()

        if fii_dii_df is not None:
            try:
                fii = float(fii_dii_df['FII_Net'].rolling(20).sum().iloc[-1])
                dii = float(fii_dii_df['DII_Net'].rolling(20).sum().iloc[-1])
                fii_cr = fii / 1e7
                pts = 30 if fii_cr > 5000 else 22 if fii_cr > 1000 else 12 if fii_cr > -1000 else 4 if fii_cr > -5000 else 0
                total += pts
                signals['fii_20d_net_cr'] = round(fii_cr, 0)
                pts = 20 if fii > 0 and dii > 0 else 14 if fii > 0 else 8 if dii > 0 else 0
                total += pts
                fii_recent = float(fii_dii_df['FII_Net'].rolling(5).sum().iloc[-1])
                fii_prior  = float(fii_dii_df['FII_Net'].rolling(5).sum().iloc[-6])
                pts = 15 if fii_recent > fii_prior * 1.2 else 10 if fii_recent > fii_prior else 5 if fii_recent > 0 else 0
                total += pts
                signals['data_source'] = 'real_fii_dii'
            except:
                fii_dii_df = None

        if fii_dii_df is None:
            r5d  = float((close.iloc[-1] / close.iloc[-5] - 1) * 100)
            v5d  = float((vix_close.iloc[-1] / vix_close.iloc[-5] - 1) * 100)
            pts = 65 if r5d > 1 and v5d < -5 else 48 if r5d > 0 and v5d < 0 else 30 if r5d > 0 else 18 if v5d < 0 else 5
            total += pts
            signals['data_source'] = 'price_proxy'

        if len(vix_close) > 20:
            slope = float((vix_close.iloc[-1] - vix_close.iloc[-20]) / 20)
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


# ── MAIN RUNNER ───────────────────────────────────────────────────────

def run():
    print(f"\n{'='*55}")
    print(f"  REGIME CLASSIFIER — {datetime.today().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*55}\n")

    end   = datetime.today()
    start = end - timedelta(days=600)

    print("Downloading Nifty 50...")
    nifty = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
    print(f"  ✓ {len(nifty)} days")

    print("Downloading India VIX...")
    vix = yf.download("^INDIAVIX", start=start, end=end, progress=False, auto_adjust=True)
    print(f"  ✓ {len(vix)} days")

    print(f"Downloading {len(NIFTY500_SAMPLE)} breadth stocks...")
    components = {}
    for t in NIFTY500_SAMPLE:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 100:
                components[t] = df
        except:
            pass
    print(f"  ✓ {len(components)} stocks loaded")

    # Score dimensions
    ts, td = TrendDimension().score(nifty)
    vs, vd = VolatilityDimension().score(nifty, vix)
    bs, bd = BreadthDimension().score(components)
    fs, fd = FlowDimension().score(nifty, vix)

    composite = ts * 0.30 + vs * 0.25 + bs * 0.25 + fs * 0.20
    code, label, action = score_to_regime(composite)

    current_price = float(nifty['Close'].iloc[-1])
    current_vix   = float(vix['Close'].iloc[-1]) if len(vix) > 0 else 0.0

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
        dimension_signals  = {'trend': td, 'volatility': vd, 'breadth': bd, 'flow': fd},
    )

    # Historical weekly scan
    print("\nBuilding 2-year history...")
    hist_records = []
    weekly_dates = nifty['2023-01-01':].resample('W').last().index

    for d in weekly_dates:
        try:
            n_slice = nifty[nifty.index <= d]
            v_slice = vix[vix.index <= d]
            if len(n_slice) < 200:
                continue

            t2, _ = TrendDimension().score(n_slice)
            v2, _ = VolatilityDimension().score(n_slice, v_slice)
            b2, _ = BreadthDimension().score(components)
            f2, _ = FlowDimension().score(n_slice, v_slice)
            c2    = t2 * 0.30 + v2 * 0.25 + b2 * 0.25 + f2 * 0.20
            rc, rl, _ = score_to_regime(c2)

            hist_records.append({
                'date':             str(d.date()),
                'nifty_price':      round(float(n_slice['Close'].iloc[-1]), 2),
                'india_vix':        round(float(v_slice['Close'].iloc[-1]) if len(v_slice) > 0 else 0, 2),
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

    # Write JSON
    with open(CURRENT_PATH, 'w') as f:
        json.dump(asdict(snap), f, indent=2, default=str)

    with open(HISTORY_PATH, 'w') as f:
        json.dump(hist_records, f, default=str)

    print(f"\n✓ Saved: {CURRENT_PATH}")
    print(f"✓ Saved: {HISTORY_PATH}")
    print(f"\n  Regime: {label} ({composite:.1f}/100)")
    print(f"  Nifty:  {current_price:,.0f}  |  VIX: {current_vix:.1f}")
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    run()
