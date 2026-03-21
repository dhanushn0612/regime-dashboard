"""
Regime Monitor — Daily Intra-Month Rebalancing Engine
=======================================================
Runs every morning before market open.
Checks three triggers and writes action_required.json if any fires.

Triggers:
  1. Regime break    — composite score crosses a regime boundary
  2. Score velocity  — drops >12 points in 5 trading days
  3. Breadth collapse — breadth dimension alone drops below 15

Usage:
    python data_pipeline/regime_monitor.py

Output:
    public/regime_monitor.json  — always written (daily status)
    public/action_required.json — written ONLY when a trigger fires
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, '..', 'public')
MONITOR_OUT  = os.path.join(OUTPUT_DIR, 'regime_monitor.json')
ACTION_OUT   = os.path.join(OUTPUT_DIR, 'action_required.json')
REGIME_CURR  = os.path.join(OUTPUT_DIR, 'regime_current.json')
REGIME_HIST  = os.path.join(OUTPUT_DIR, 'regime_history.json')
PORT_CURR    = os.path.join(OUTPUT_DIR, 'portfolio_current.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── REGIME THRESHOLDS ─────────────────────────────────────────────────
REGIME_BOUNDARIES = {4: 75, 3: 55, 2: 40, 1: 20, 0: 0}
REGIME_LABELS     = {4:'Strong Bull', 3:'Mild Bull', 2:'Neutral', 1:'Mild Bear', 0:'Strong Bear'}

# ── TARGET EQUITY BY SCORE (continuous, replaces flat allocation) ─────
def score_to_equity(score: float, regime_code: int) -> float:
    """
    Score-proportional equity allocation within each regime.
    Replaces the flat allocation table with a continuous function.
    Neutral regime gets sub-banding based on exact score.
    """
    if regime_code == 0:   return 0.00
    if regime_code == 1:   return 0.15
    if regime_code == 4:   return 0.90
    if regime_code == 3:
        # Mild Bull: 75-80% proportional to score 55-75
        return round(0.75 + (score - 55) / (75 - 55) * 0.05, 2)
    if regime_code == 2:
        # Neutral sub-banding based on exact score
        if score < 45:   return 0.55
        elif score < 50: return 0.62
        elif score < 55: return 0.68
        else:            return 0.73
    return 0.45


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


# ── COMPUTE CURRENT REGIME (daily) ───────────────────────────────────
def compute_daily_regime() -> dict:
    """
    Compute today's composite score using live data.
    Simplified version for speed — same dimensions as full classifier.
    """
    end   = datetime.today()
    start = end - timedelta(days=400)

    # Nifty
    nifty = flatten(yf.download("^NSEI", start=start, end=end,
                                 progress=False, auto_adjust=True))
    nc    = get_close(nifty)

    # VIX
    vix_df = flatten(yf.download("^INDIAVIX", start=start, end=end,
                                  progress=False, auto_adjust=True))
    vc     = get_close(vix_df) if len(vix_df) > 0 else pd.Series(dtype=float)

    if len(nc) < 50:
        return {'error': 'insufficient data'}

    curr = s(nc.iloc[-1])

    # Trend
    trend = 0
    try:
        sma50  = s(nc.rolling(50).mean().iloc[-1])
        dist50 = (curr - sma50) / sma50 * 100
        trend += 20 if dist50 > 3 else 14 if dist50 > 0 else 6 if dist50 > -3 else 0
    except: pass
    try:
        sma200 = s(nc.rolling(200).mean().iloc[-1])
        if len(nc) >= 200:
            dist200 = (curr - sma200) / sma200 * 100
            trend += 20 if dist200 > 5 else 13 if dist200 > 0 else 5 if dist200 > -5 else 0
            sma50v  = s(nc.rolling(50).mean().iloc[-1])
            trend  += 15 if sma50v > sma200 else 0
    except: pass
    try:
        if len(nc) > 21:
            roc1m = (curr / s(nc.iloc[-21]) - 1) * 100
            trend += 10 if roc1m > 3 else 6 if roc1m > 0 else 2 if roc1m > -3 else 0
        if len(nc) > 63:
            roc3m = (curr / s(nc.iloc[-63]) - 1) * 100
            trend += 10 if roc3m > 7 else 6 if roc3m > 0 else 2 if roc3m > -7 else 0
        high52 = s(nc.rolling(min(252, len(nc))).max().iloc[-1])
        dist52 = (curr / high52 - 1) * 100
        trend += 15 if dist52 > -5 else 10 if dist52 > -10 else 4 if dist52 > -20 else 0
    except: pass
    trend = min(100, max(0, trend))

    # Volatility
    vola = 50
    try:
        if len(vc) >= 5:
            cv   = s(vc.iloc[-1])
            vola = 35 if cv < 13 else 28 if cv < 16 else 18 if cv < 20 else 10 if cv < 25 else 4 if cv < 30 else 0
        if len(vc) >= 20:
            vchg = (s(vc.iloc[-1]) / s(vc.iloc[-20]) - 1) * 100
            vola += 20 if vchg < -15 else 15 if vchg < -5 else 10 if vchg < 5 else 4 if vchg < 15 else 0
        rets = nc.pct_change().dropna()
        if len(rets) >= 60:
            rv20  = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
            rv60  = s(rets.rolling(60).std().iloc[-1]) * np.sqrt(252) * 100
            ratio = rv20 / rv60 if rv60 > 0 else 1.0
            vola += 25 if ratio < 0.7 else 18 if ratio < 0.9 else 12 if ratio < 1.1 else 5 if ratio < 1.4 else 0
        if len(vc) >= 10:
            vov   = s(vc.rolling(10).std().iloc[-1])
            vola += 20 if vov < 1.0 else 14 if vov < 2.0 else 7 if vov < 3.5 else 0
        vola = min(100, max(0, vola))
    except: pass

    # Breadth (20 liquid stocks)
    BREADTH_STOCKS = [
        "HDFCBANK.NS","ICICIBANK.NS","TCS.NS","INFY.NS","RELIANCE.NS",
        "HINDUNILVR.NS","ITC.NS","LT.NS","AXISBANK.NS","SBIN.NS",
        "SUNPHARMA.NS","MARUTI.NS","TITAN.NS","BAJFINANCE.NS","WIPRO.NS",
        "DRREDDY.NS","NTPC.NS","POWERGRID.NS","ASIANPAINT.NS","KOTAKBANK.NS",
    ]
    breadth = 50
    above50 = []
    above200 = []
    try:
        for ticker in BREADTH_STOCKS:
            try:
                df = flatten(yf.download(ticker, start=start, end=end,
                                          progress=False, auto_adjust=True))
                c  = get_close(df)
                if len(c) >= 50:
                    above50.append(s(c.iloc[-1]) > s(c.rolling(50).mean().iloc[-1]))
                if len(c) >= 200:
                    above200.append(s(c.iloc[-1]) > s(c.rolling(200).mean().iloc[-1]))
            except: pass
        if above50:
            p50     = sum(above50) / len(above50) * 100
            breadth = 35 if p50 > 70 else 25 if p50 > 55 else 15 if p50 > 40 else 7 if p50 > 30 else 0
        if above200:
            p200     = sum(above200) / len(above200) * 100
            breadth += 35 if p200 > 65 else 25 if p200 > 50 else 15 if p200 > 35 else 7 if p200 > 25 else 0
        breadth = min(100, max(0, breadth))
    except: pass

    # Flow (VIX slope proxy)
    flow = 40
    try:
        if len(vc) >= 5 and len(nc) >= 5:
            r5 = (curr / s(nc.iloc[-5]) - 1) * 100
            v5 = (s(vc.iloc[-1]) / s(vc.iloc[-5]) - 1) * 100
            flow = 65 if r5 > 1 and v5 < -5 else 48 if r5 > 0 and v5 < 0 else 30 if r5 > 0 else 18 if v5 < 0 else 5
        if len(vc) >= 20:
            slope = (s(vc.iloc[-1]) - s(vc.iloc[-20])) / 20
            flow += 35 if slope < -0.2 else 25 if slope < 0 else 15 if slope < 0.2 else 6 if slope < 0.5 else 0
        flow = min(100, max(0, flow))
    except: pass

    composite = trend * 0.30 + vola * 0.25 + breadth * 0.25 + flow * 0.20
    code = 4 if composite >= 75 else 3 if composite >= 55 else 2 if composite >= 40 else 1 if composite >= 20 else 0

    return {
        'date':      datetime.today().strftime('%Y-%m-%d'),
        'score':     round(composite, 1),
        'code':      code,
        'label':     REGIME_LABELS[code],
        'trend':     round(trend, 1),
        'vola':      round(vola, 1),
        'breadth':   round(breadth, 1),
        'flow':      round(flow, 1),
        'nifty_price': round(curr, 1),
    }


# ── TRIGGER 1: REGIME BREAK ───────────────────────────────────────────
def check_regime_break(current_score: float, current_code: int,
                        prev_code: int) -> dict:
    """Fires when the composite crosses a regime boundary."""
    if current_code != prev_code:
        direction = 'down' if current_code < prev_code else 'up'
        severity  = 'CRITICAL' if abs(current_code - prev_code) >= 2 else 'ALERT'
        return {
            'triggered':   True,
            'trigger':     'regime_break',
            'severity':    severity,
            'direction':   direction,
            'from_regime': REGIME_LABELS[prev_code],
            'to_regime':   REGIME_LABELS[current_code],
            'message':     f"Regime crossed from {REGIME_LABELS[prev_code]} to {REGIME_LABELS[current_code]}",
            'target_equity': score_to_equity(current_score, current_code),
        }
    return {'triggered': False, 'trigger': 'regime_break'}


# ── TRIGGER 2: SCORE VELOCITY ─────────────────────────────────────────
def check_score_velocity(current_score: float, history: list) -> dict:
    """Fires when score drops >12 points in 5 periods."""
    if len(history) < 5:
        return {'triggered': False, 'trigger': 'score_velocity'}

    scores_5d = [h.get('composite_score', 50) for h in history[-5:]]
    score_5d_ago = scores_5d[0]
    drop = score_5d_ago - current_score  # Positive = dropped

    if drop > 12:
        current_code = 4 if current_score >= 75 else 3 if current_score >= 55 else 2 if current_score >= 40 else 1 if current_score >= 20 else 0
        # De-risk to next lower regime
        target_code  = max(0, current_code - 1)
        target_equity = score_to_equity(current_score, target_code)
        return {
            'triggered':    True,
            'trigger':      'score_velocity',
            'severity':     'ALERT',
            'score_now':    round(current_score, 1),
            'score_5d_ago': round(score_5d_ago, 1),
            'drop':         round(drop, 1),
            'message':      f"Score dropped {drop:.1f} pts in 5 periods ({score_5d_ago:.0f} → {current_score:.0f})",
            'target_equity': target_equity,
        }
    return {'triggered': False, 'trigger': 'score_velocity', 'drop': round(drop, 1)}


# ── TRIGGER 3: BREADTH COLLAPSE ───────────────────────────────────────
def check_breadth_collapse(breadth_score: float,
                            current_equity: float) -> dict:
    """Fires when breadth alone drops below 15 — leading indicator of crashes."""
    if breadth_score < 15:
        target_equity = round(current_equity * 0.50, 2)
        return {
            'triggered':     True,
            'trigger':       'breadth_collapse',
            'severity':      'CRITICAL',
            'breadth_score': round(breadth_score, 1),
            'message':       f"Breadth collapsed to {breadth_score:.0f}/100 — internals failing",
            'target_equity': target_equity,
            'action':        'Reduce equity by 50% immediately',
        }
    return {'triggered': False, 'trigger': 'breadth_collapse', 'breadth_score': round(breadth_score, 1)}


# ── BUILD ACTION SIGNAL ───────────────────────────────────────────────
def build_action_signal(triggers: list, current: dict,
                         portfolio: dict) -> dict:
    """
    Combine all triggered signals into a single action recommendation.
    Most severe trigger governs the target allocation.
    """
    triggered = [t for t in triggers if t.get('triggered')]
    if not triggered:
        return None

    # Most severe first
    severity_order = {'CRITICAL': 0, 'ALERT': 1, 'WARN': 2}
    triggered.sort(key=lambda x: severity_order.get(x.get('severity', 'WARN'), 2))
    worst = triggered[0]

    current_equity = portfolio.get('equity_allocation', 0)
    target_equity  = worst.get('target_equity', current_equity * 0.7)

    positions = portfolio.get('positions', [])

    # Which positions to exit first (highest vol, worst performers)
    exit_first = []
    keep        = []
    for pos in positions:
        # Exit high-vol or underperformers first
        ann_vol = pos.get('ann_vol_pct', 30)
        if ann_vol > 35 or pos.get('expected_ret', 0) < 0:
            exit_first.append(pos['ticker'])
        else:
            keep.append(pos['ticker'])

    return {
        'date':            datetime.today().strftime('%Y-%m-%d %H:%M'),
        'action_required': True,
        'triggers_fired':  [t['trigger'] for t in triggered],
        'worst_severity':  worst.get('severity'),
        'primary_message': worst.get('message'),
        'current_equity':  round(current_equity * 100, 1),
        'target_equity':   round(target_equity * 100, 1),
        'equity_change':   round((target_equity - current_equity) * 100, 1),
        'exit_first':      exit_first[:5],
        'keep':            keep[:5],
        'trigger_details': triggered,
        'instructions': [
            f"Reduce equity from {current_equity*100:.0f}% to {target_equity*100:.0f}%",
            f"Exit these positions first: {', '.join(exit_first[:3]) if exit_first else 'none identified'}",
            f"Keep defensive positions: {', '.join(keep[:3]) if keep else 'all if forced'}",
            "Review and execute before market open",
            "Document the action in your trading log",
        ]
    }


# ── MAIN ──────────────────────────────────────────────────────────────
def run():
    print("\n" + "="*55)
    print("  REGIME MONITOR")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*55 + "\n")

    # Load existing regime data
    prev_code  = 2
    prev_score = 50.0
    history    = []

    if os.path.exists(REGIME_CURR):
        with open(REGIME_CURR) as f:
            reg = json.load(f)
        prev_code  = reg.get('regime_code', 2)
        prev_score = reg.get('composite_score', 50.0)

    if os.path.exists(REGIME_HIST):
        with open(REGIME_HIST) as f:
            history = json.load(f)

    # Load portfolio
    portfolio = {}
    if os.path.exists(PORT_CURR):
        with open(PORT_CURR) as f:
            portfolio = json.load(f)

    current_equity = portfolio.get('equity_allocation', 0)

    # Compute today's regime
    print("Computing daily regime score...")
    current = compute_daily_regime()

    if 'error' in current:
        print(f"  Error: {current['error']}")
        return

    current_score = current['score']
    current_code  = current['code']
    breadth_score = current['breadth']

    print(f"  Score:   {current_score}/100")
    print(f"  Regime:  {current['label']}")
    print(f"  Trend:   {current['trend']}  Vola: {current['vola']}  Breadth: {current['breadth']}  Flow: {current['flow']}")
    print(f"  Nifty:   {current['nifty_price']}")
    print(f"  Change:  {current_score - prev_score:+.1f} pts from last reading")

    # Run triggers
    print("\nChecking triggers...")
    t1 = check_regime_break(current_score, current_code, prev_code)
    t2 = check_score_velocity(current_score, history)
    t3 = check_breadth_collapse(breadth_score, current_equity)

    triggers = [t1, t2, t3]
    for t in triggers:
        status = "🔴 FIRED" if t['triggered'] else "✅ clear"
        print(f"  {t['trigger']:<22}: {status}" +
              (f" — {t.get('message','')}" if t['triggered'] else ""))

    # Target allocation today
    target_equity = score_to_equity(current_score, current_code)
    print(f"\n  Current equity: {current_equity*100:.0f}%")
    print(f"  Target equity:  {target_equity*100:.0f}%")

    # Build action signal if needed
    action = build_action_signal(triggers, current, portfolio)

    # Save monitor output (always)
    monitor_data = {
        'date':           current['date'],
        'score':          current_score,
        'regime_code':    current_code,
        'regime_label':   current['label'],
        'trend':          current['trend'],
        'vola':           current['vola'],
        'breadth':        current['breadth'],
        'flow':           current['flow'],
        'nifty_price':    current['nifty_price'],
        'score_change':   round(current_score - prev_score, 1),
        'prev_score':     prev_score,
        'prev_code':      prev_code,
        'target_equity':  target_equity,
        'current_equity': current_equity,
        'triggers': {
            'regime_break':      t1,
            'score_velocity':    t2,
            'breadth_collapse':  t3,
        },
        'action_required': action is not None,
    }

    with open(MONITOR_OUT, 'w') as f:
        json.dump(monitor_data, f, indent=2, default=str)

    # Save action signal if triggered
    if action:
        with open(ACTION_OUT, 'w') as f:
            json.dump(action, f, indent=2, default=str)
        print(f"\n  ⚠️  ACTION REQUIRED — {action['worst_severity']}")
        print(f"  Reduce equity: {action['current_equity']}% → {action['target_equity']}%")
        print(f"  Exit first: {', '.join(action['exit_first'][:3]) or 'see action_required.json'}")
        print(f"\n  Saved: {ACTION_OUT}")
    else:
        # Clear old action file if no triggers
        if os.path.exists(ACTION_OUT):
            old = json.load(open(ACTION_OUT))
            old['cleared'] = datetime.today().strftime('%Y-%m-%d')
            old['action_required'] = False
            json.dump(old, open(ACTION_OUT, 'w'), indent=2, default=str)
        print(f"\n  ✅ No action required")

    print(f"\n  Saved: {MONITOR_OUT}")
    print(f"  Run this daily before 9:15 AM IST\n")


if __name__ == "__main__":
    run()
