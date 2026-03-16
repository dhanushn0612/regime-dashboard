"""
Risk Management — Layer 5
===========================
Industry-standard systematic risk management.
Five independent risk rules, each catching different failure modes.
Auto position size reduction when rules trigger.

Rules (industry standard):
  1. Portfolio Drawdown     — 15% from high-water mark triggers 50% de-risk
  2. Single Stock Stop Loss — 12% loss from entry triggers full exit
  3. Factor Concentration   — >60% portfolio in one factor triggers rebalance
  4. Correlation Spike      — avg pairwise correlation >0.75 triggers reduction
  5. Regime Deterioration   — regime score drops >20pts in 5 days triggers alert

ML Addition:
  Isolation Forest anomaly detector flags when current market conditions
  are statistically unusual vs historical patterns — early warning system.

Usage:
    python data_pipeline/risk_management.py

Output:
    public/risk_current.json  — current risk status and alerts
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

warnings.filterwarnings('ignore')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, '..', 'public')
RISK_CURR   = os.path.join(OUTPUT_DIR, 'risk_current.json')
PORT_CURR   = os.path.join(OUTPUT_DIR, 'portfolio_current.json')
REGIME_CURR = os.path.join(OUTPUT_DIR, 'regime_current.json')
REGIME_HIST = os.path.join(OUTPUT_DIR, 'regime_history.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── RISK THRESHOLDS (industry standard) ──────────────────────────────
THRESHOLDS = {
    'max_drawdown':          0.15,    # 15% portfolio drawdown → de-risk 50%
    'stock_stop_loss':       0.12,    # 12% single stock loss → exit position
    'factor_concentration':  0.60,    # 60% in one factor → rebalance
    'correlation_spike':     0.75,    # Avg pairwise corr > 0.75 → reduce 30%
    'regime_drop_5d':        20.0,    # Regime score drops 20pts in 5 days → alert
    'regime_drop_critical':  15.0,    # Regime drops below 15 → emergency de-risk
}

# ── SEVERITY LEVELS ───────────────────────────────────────────────────
SEVERITY = {
    'INFO':     {'color': '#00c896', 'action': 'Monitor',              'size_mult': 1.00},
    'WARN':     {'color': '#ffd166', 'action': 'Review positions',     'size_mult': 0.80},
    'ALERT':    {'color': '#ff6b6b', 'action': 'Reduce 30% exposure',  'size_mult': 0.70},
    'CRITICAL': {'color': '#ff2d55', 'action': 'Reduce 50% exposure',  'size_mult': 0.50},
}


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


# ── RULE 1: PORTFOLIO DRAWDOWN ────────────────────────────────────────
def check_portfolio_drawdown(portfolio_data: dict,
                              price_df: pd.DataFrame) -> dict:
    """
    Measure current portfolio drawdown from high-water mark.

    High-water mark: highest portfolio value ever achieved.
    Drawdown: (current value - HWM) / HWM

    Industry standard: 15% drawdown → mandatory de-risking.
    This is the most important risk rule because it prevents
    the common mistake of riding a losing portfolio into catastrophe.
    """
    positions = portfolio_data.get('positions', [])
    if not positions:
        return {
            'rule': 'portfolio_drawdown',
            'triggered': False,
            'severity': 'INFO',
            'message': 'No positions — drawdown not applicable',
            'drawdown_pct': 0.0,
        }

    # Compute portfolio daily returns from position weights
    tickers  = [p['ticker'] for p in positions]
    weights  = [p['target_weight'] for p in positions]
    eq_alloc = portfolio_data.get('equity_allocation', 1.0)

    available = [(t, w) for t, w in zip(tickers, weights) if t in price_df.columns]
    if not available:
        return {
            'rule': 'portfolio_drawdown',
            'triggered': False,
            'severity': 'INFO',
            'message': 'Price data unavailable for positions',
            'drawdown_pct': 0.0,
        }

    # Build weighted return series (last 60 days)
    port_rets = pd.Series(0.0, index=price_df.index[-60:])
    for ticker, weight in available:
        close = get_close(price_df[[ticker]]) if ticker in price_df.columns else None
        if close is not None:
            rets = close.pct_change().fillna(0)
            port_rets = port_rets.add(rets.reindex(port_rets.index, fill_value=0) * weight, fill_value=0)

    # Cash drag (cash earns 0 in this model)
    port_rets = port_rets * eq_alloc

    # Cumulative portfolio value
    cum_value = (1 + port_rets).cumprod()
    hwm       = cum_value.cummax()
    drawdown  = (cum_value - hwm) / hwm
    current_dd = float(drawdown.iloc[-1]) * 100  # As percentage

    max_dd = float(drawdown.min()) * 100
    threshold = THRESHOLDS['max_drawdown'] * 100

    triggered = current_dd < -threshold
    severity  = 'CRITICAL' if current_dd < -threshold else \
                'ALERT'    if current_dd < -threshold * 0.7 else \
                'WARN'     if current_dd < -threshold * 0.4 else 'INFO'

    return {
        'rule':           'portfolio_drawdown',
        'triggered':      triggered,
        'severity':       severity,
        'current_dd_pct': round(current_dd, 2),
        'max_dd_pct':     round(max_dd, 2),
        'threshold_pct':  -threshold,
        'message':        f"Portfolio drawdown: {current_dd:.1f}% (limit: -{threshold:.0f}%)",
        'action':         SEVERITY[severity]['action'],
        'size_multiplier':SEVERITY[severity]['size_mult'] if triggered else 1.0,
    }


# ── RULE 2: SINGLE STOCK STOP LOSS ───────────────────────────────────
def check_stock_stop_losses(portfolio_data: dict,
                             price_df: pd.DataFrame) -> dict:
    """
    Check if any individual stock has breached its stop loss.

    Stop loss: 12% decline from entry price.
    Entry price: approximated as price when position was last initiated
    (using portfolio construction date as proxy).

    Industry practice: individual position stops prevent a single
    stock blowup from damaging the whole portfolio. Without stops,
    one fraud or earnings disaster can wipe months of gains.
    """
    positions  = portfolio_data.get('positions', [])
    port_date  = portfolio_data.get('date', '')
    breaches   = []
    clean      = []

    for pos in positions:
        ticker = pos['ticker']
        if ticker not in price_df.columns:
            continue

        close = get_close(price_df[[ticker]])

        # Entry price: price on portfolio construction date
        # If date not available, use 21-day-ago price as proxy
        try:
            entry_price = s(close.iloc[-21])   # 1 month ago as proxy
        except Exception:
            entry_price = s(close.iloc[0])

        current_price = s(close.iloc[-1])
        loss_pct = (current_price / entry_price - 1) * 100

        threshold = -THRESHOLDS['stock_stop_loss'] * 100

        if loss_pct < threshold:
            breaches.append({
                'ticker':         ticker,
                'name':           pos['name'],
                'loss_pct':       round(loss_pct, 2),
                'threshold_pct':  threshold,
                'action':         'EXIT POSITION',
            })
        else:
            clean.append({
                'ticker':   ticker,
                'loss_pct': round(loss_pct, 2),
            })

    triggered = len(breaches) > 0
    severity  = 'CRITICAL' if len(breaches) >= 2 else 'ALERT' if triggered else 'INFO'

    return {
        'rule':            'stock_stop_loss',
        'triggered':       triggered,
        'severity':        severity,
        'breaches':        breaches,
        'clean_positions': len(clean),
        'threshold_pct':   -THRESHOLDS['stock_stop_loss'] * 100,
        'message':         f"{len(breaches)} position(s) breached {THRESHOLDS['stock_stop_loss']*100:.0f}% stop loss" if triggered else f"All {len(clean)} positions within stop loss limits",
        'action':          SEVERITY[severity]['action'],
        'size_multiplier': SEVERITY[severity]['size_mult'] if triggered else 1.0,
    }


# ── RULE 3: FACTOR CONCENTRATION ──────────────────────────────────────
def check_factor_concentration(portfolio_data: dict,
                                screener_path: str = None) -> dict:
    """
    Check if portfolio is overly concentrated in a single factor.

    Factor concentration risk: if 70% of your portfolio's return
    is driven by momentum, then a momentum crash (common in crises)
    wipes 70% of your gains simultaneously.

    Detection method: look at factor scores of held positions.
    If the average of any single factor across positions >> others,
    flag as concentrated.
    """
    positions = portfolio_data.get('positions', [])
    if not positions:
        return {
            'rule': 'factor_concentration',
            'triggered': False,
            'severity': 'INFO',
            'message': 'No positions',
            'factor_scores': {},
        }

    # Compute weighted average factor scores
    total_weight = sum(p.get('target_weight', 0) for p in positions)
    if total_weight == 0:
        return {'rule': 'factor_concentration', 'triggered': False, 'severity': 'INFO', 'message': 'Zero weight', 'factor_scores': {}}

    factor_avgs = {}
    for factor in ['f_momentum', 'f_quality', 'f_lowvol', 'f_earnings']:
        weighted_sum = sum(
            p.get(factor, 50) * p.get('target_weight', 0)
            for p in positions
        )
        factor_avgs[factor] = weighted_sum / total_weight

    # Normalise to 0-1 (scores are already 0-100, normalise by sum)
    total_score = sum(factor_avgs.values())
    if total_score == 0:
        factor_avgs = {k: 0.25 for k in factor_avgs}
    else:
        factor_avgs = {k: v / total_score for k, v in factor_avgs.items()}

    max_factor       = max(factor_avgs, key=factor_avgs.get)
    max_concentration = factor_avgs[max_factor]
    threshold         = THRESHOLDS['factor_concentration']

    triggered = max_concentration > threshold
    severity  = 'ALERT' if max_concentration > threshold else \
                'WARN'  if max_concentration > threshold * 0.85 else 'INFO'

    return {
        'rule':               'factor_concentration',
        'triggered':          triggered,
        'severity':           severity,
        'dominant_factor':    max_factor.replace('f_', ''),
        'concentration':      round(max_concentration * 100, 1),
        'threshold_pct':      threshold * 100,
        'factor_scores':      {k.replace('f_',''): round(v*100, 1) for k, v in factor_avgs.items()},
        'message':            f"Factor concentration: {max_factor.replace('f_','')} at {max_concentration*100:.0f}% (limit: {threshold*100:.0f}%)" if triggered else f"Factor concentration balanced — max {max_concentration*100:.0f}% in {max_factor.replace('f_','')}",
        'action':             SEVERITY[severity]['action'],
        'size_multiplier':    SEVERITY[severity]['size_mult'] if triggered else 1.0,
    }


# ── RULE 4: CORRELATION SPIKE ─────────────────────────────────────────
def check_correlation_spike(portfolio_data: dict,
                             price_df: pd.DataFrame) -> dict:
    """
    Detect when portfolio stocks become highly correlated.

    Correlation spike = diversification failure.
    When all stocks move together (corr > 0.75), your 10 stocks
    behave like 1 stock. Your Effective N drops from 10 to ~2.

    This happens during market stress — exactly when you need
    diversification the most. Early detection allows position
    reduction before the correlated drawdown arrives.

    Measurement: average pairwise Pearson correlation over 20 days.
    """
    positions = portfolio_data.get('positions', [])
    tickers   = [p['ticker'] for p in positions if p['ticker'] in price_df.columns]

    if len(tickers) < 2:
        return {
            'rule': 'correlation_spike',
            'triggered': False,
            'severity': 'INFO',
            'message': 'Fewer than 2 positions — correlation not applicable',
            'avg_correlation': 0.0,
        }

    # 20-day rolling correlation
    returns = price_df[tickers].pct_change().tail(20).dropna()
    if len(returns) < 10:
        return {'rule': 'correlation_spike', 'triggered': False, 'severity': 'INFO', 'message': 'Insufficient data', 'avg_correlation': 0.0}

    corr_matrix  = returns.corr()
    n = len(tickers)

    # Average of upper triangle (exclude diagonal)
    upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
    avg_corr  = float(np.mean(upper_tri))

    threshold = THRESHOLDS['correlation_spike']
    triggered = avg_corr > threshold
    severity  = 'CRITICAL' if avg_corr > threshold * 1.1 else \
                'ALERT'    if avg_corr > threshold else \
                'WARN'     if avg_corr > threshold * 0.85 else 'INFO'

    # Find most correlated pair
    max_corr_val = float(np.max(upper_tri)) if len(upper_tri) > 0 else 0
    max_idx = np.unravel_index(np.argmax(corr_matrix.values - np.eye(n)), (n, n))
    most_correlated = f"{tickers[max_idx[0]].replace('.NS','')} / {tickers[max_idx[1]].replace('.NS','')}"

    return {
        'rule':              'correlation_spike',
        'triggered':         triggered,
        'severity':          severity,
        'avg_correlation':   round(avg_corr, 3),
        'max_correlation':   round(max_corr_val, 3),
        'most_correlated':   most_correlated,
        'threshold':         threshold,
        'effective_n':       round(1 / (avg_corr + 0.01), 1),
        'message':           f"Avg correlation: {avg_corr:.2f} (limit: {threshold:.2f}) — diversification {'FAILING' if triggered else 'OK'}",
        'action':            SEVERITY[severity]['action'],
        'size_multiplier':   SEVERITY[severity]['size_mult'] if triggered else 1.0,
    }


# ── RULE 5: REGIME DETERIORATION ──────────────────────────────────────
def check_regime_deterioration(regime_current: dict,
                                regime_history: list) -> dict:
    """
    Detect rapid regime score deterioration.

    Why this matters: the regime score is a weekly signal.
    A sudden 20-point drop in 5 days means something significant
    changed in the market before the weekly rebalance catches it.

    This is the early warning system — it fires BEFORE the regime
    crosses into a lower bucket, giving you time to reduce exposure
    proactively rather than reactively.

    Critical threshold: score drops below 15 (approaching Strong Bear).
    """
    current_score = regime_current.get('composite_score', 50)
    current_code  = regime_current.get('regime_code', 2)

    if not regime_history or len(regime_history) < 5:
        return {
            'rule': 'regime_deterioration',
            'triggered': False,
            'severity': 'INFO',
            'message': 'Insufficient regime history',
            'score_change_5d': 0.0,
        }

    # Get scores from last 5 periods
    recent_scores = [r.get('composite_score', 50) for r in regime_history[-5:]]
    score_5d_ago  = recent_scores[0]
    score_change  = current_score - score_5d_ago

    # Check critical threshold
    critical = current_score < THRESHOLDS['regime_drop_critical']
    rapid    = score_change < -THRESHOLDS['regime_drop_5d']

    triggered = critical or rapid
    severity  = 'CRITICAL' if critical else \
                'ALERT'    if rapid else \
                'WARN'     if score_change < -10 else 'INFO'

    return {
        'rule':             'regime_deterioration',
        'triggered':        triggered,
        'severity':         severity,
        'current_score':    round(current_score, 1),
        'score_5d_ago':     round(score_5d_ago, 1),
        'score_change_5d':  round(score_change, 1),
        'is_critical':      critical,
        'is_rapid_drop':    rapid,
        'message':          f"Regime: {current_score:.1f} ({score_change:+.1f} in 5 periods)" + (" — CRITICAL LEVEL" if critical else " — RAPID DETERIORATION" if rapid else ""),
        'action':           SEVERITY[severity]['action'],
        'size_multiplier':  SEVERITY[severity]['size_mult'] if triggered else 1.0,
    }


# ── ML RULE: ISOLATION FOREST ANOMALY DETECTION ───────────────────────
def check_market_anomaly(regime_current: dict,
                          regime_history: list) -> dict:
    """
    Isolation Forest anomaly detection.

    Isolation Forest is an unsupervised ML model that identifies
    data points that are 'isolated' from the rest of the distribution.
    Points that are easy to isolate (require few splits) are anomalies.

    For market regimes: we train on the 2-year history of the 4 dimension
    scores. If today's scores are statistically unusual compared to
    the historical distribution, the model flags it as an anomaly.

    This catches tail events that rule-based systems miss — situations
    where no single threshold is breached but the combination of signals
    is historically unprecedented.

    Anomaly score: -1 = anomaly, 1 = normal
    Contamination: 0.05 = expect 5% of periods to be anomalous
    """
    try:
        from sklearn.ensemble import IsolationForest
        HAS_IF = True
    except ImportError:
        HAS_IF = False

    if not HAS_IF or not regime_history or len(regime_history) < 20:
        return {
            'rule': 'market_anomaly',
            'triggered': False,
            'severity': 'INFO',
            'message': 'Isolation Forest requires sklearn and 20+ history points',
            'anomaly_score': 0.0,
        }

    # Build feature matrix from history
    features = []
    for r in regime_history:
        features.append([
            r.get('trend_score', 50),
            r.get('volatility_score', 50),
            r.get('breadth_score', 50),
            r.get('flow_score', 50),
            r.get('composite_score', 50),
        ])

    X_hist = np.array(features[:-1])  # Training: all but last
    X_curr = np.array([features[-1]])  # Current period

    # Train Isolation Forest
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.05,   # Expect 5% anomalies
        random_state=42,
    )
    clf.fit(X_hist)

    # Score current period
    prediction = clf.predict(X_curr)[0]    # -1 = anomaly, 1 = normal
    score      = clf.score_samples(X_curr)[0]  # Lower = more anomalous

    is_anomaly = prediction == -1
    severity   = 'ALERT' if is_anomaly else 'INFO'

    # Find what's unusual by comparing to historical distribution
    curr_values = features[-1]
    hist_means  = np.mean(X_hist, axis=0)
    hist_stds   = np.std(X_hist, axis=0) + 0.01
    z_scores    = (np.array(curr_values) - hist_means) / hist_stds

    dim_names = ['trend', 'volatility', 'breadth', 'flow', 'composite']
    most_unusual = dim_names[np.argmax(np.abs(z_scores))]
    most_unusual_z = float(z_scores[np.argmax(np.abs(z_scores))])

    return {
        'rule':           'market_anomaly',
        'triggered':      is_anomaly,
        'severity':       severity,
        'anomaly_score':  round(float(score), 4),
        'is_anomaly':     is_anomaly,
        'most_unusual_dim': most_unusual,
        'z_score':        round(most_unusual_z, 2),
        'z_scores':       {d: round(float(z), 2) for d, z in zip(dim_names, z_scores)},
        'message':        f"Market {'ANOMALY DETECTED' if is_anomaly else 'normal'} — {most_unusual} dimension {most_unusual_z:+.1f} std devs from historical mean",
        'action':         SEVERITY[severity]['action'],
        'size_multiplier':0.70 if is_anomaly else 1.0,
    }


# ── AGGREGATE RISK SCORE ──────────────────────────────────────────────
def compute_aggregate_risk(rules: list) -> dict:
    """
    Combine all rule outputs into a single risk score and recommendation.

    Risk score: 0-100 (100 = maximum risk, 0 = no risk)
    Overall recommendation: most severe triggered rule governs.
    """
    severity_order = ['INFO', 'WARN', 'ALERT', 'CRITICAL']
    severity_scores = {'INFO': 0, 'WARN': 25, 'ALERT': 60, 'CRITICAL': 100}

    triggered_rules  = [r for r in rules if r.get('triggered', False)]
    max_severity     = max((r['severity'] for r in rules), key=lambda x: severity_order.index(x))
    risk_score       = max(severity_scores[r['severity']] for r in rules)
    min_size_mult    = min(r.get('size_multiplier', 1.0) for r in rules)

    if max_severity == 'INFO':
        overall = "All clear — no risk rules triggered"
        color   = "#00c896"
    elif max_severity == 'WARN':
        overall = "Caution — monitor positions closely"
        color   = "#ffd166"
    elif max_severity == 'ALERT':
        overall = "Alert — reduce exposure by 30%"
        color   = "#ff6b6b"
    else:
        overall = "Critical — reduce exposure by 50% immediately"
        color   = "#ff2d55"

    return {
        'risk_score':          risk_score,
        'max_severity':        max_severity,
        'color':               color,
        'triggered_count':     len(triggered_rules),
        'overall_message':     overall,
        'position_size_mult':  round(min_size_mult, 2),
        'adjusted_equity_pct': None,  # Set below
    }


# ── MAIN ──────────────────────────────────────────────────────────────
def run():
    print("\n" + "="*60)
    print("  RISK MANAGEMENT — LAYER 5")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*60 + "\n")

    # Load inputs
    regime_current = {}
    regime_history = []
    portfolio_data = {}

    if os.path.exists(REGIME_CURR):
        with open(REGIME_CURR) as f:
            regime_current = json.load(f)

    if os.path.exists(REGIME_HIST):
        with open(REGIME_HIST) as f:
            regime_history = json.load(f)

    if os.path.exists(PORT_CURR):
        with open(PORT_CURR) as f:
            portfolio_data = json.load(f)

    regime_code  = regime_current.get('regime_code', 2)
    regime_label = regime_current.get('regime_label', 'Neutral')
    comp_score   = regime_current.get('composite_score', 50)

    print(f"  Regime: {regime_label} ({comp_score}/100)")
    print(f"  Positions: {len(portfolio_data.get('positions', []))}")

    # Download price data for positions
    tickers = [p['ticker'] for p in portfolio_data.get('positions', [])]
    price_df = pd.DataFrame()

    if tickers:
        print("\nDownloading price data for risk checks...")
        end   = datetime.today()
        start = end - timedelta(days=90)
        prices = {}
        for ticker in tickers:
            try:
                df = flatten(yf.download(ticker, start=start, end=end,
                                         progress=False, auto_adjust=True))
                if len(df) > 10:
                    prices[ticker] = get_close(df)
            except Exception:
                pass
        if prices:
            price_df = pd.DataFrame(prices)

    # ── RUN ALL FIVE RULES ─────────────────────────────────────────────
    print("\nRunning risk rules...")
    rules = []

    r1 = check_portfolio_drawdown(portfolio_data, price_df)
    rules.append(r1)
    print(f"  Rule 1 — Portfolio Drawdown:    [{r1['severity']}] {r1['message']}")

    r2 = check_stock_stop_losses(portfolio_data, price_df)
    rules.append(r2)
    print(f"  Rule 2 — Stock Stop Loss:       [{r2['severity']}] {r2['message']}")

    r3 = check_factor_concentration(portfolio_data)
    rules.append(r3)
    print(f"  Rule 3 — Factor Concentration:  [{r3['severity']}] {r3['message']}")

    r4 = check_correlation_spike(portfolio_data, price_df)
    rules.append(r4)
    print(f"  Rule 4 — Correlation Spike:     [{r4['severity']}] {r4['message']}")

    r5 = check_regime_deterioration(regime_current, regime_history)
    rules.append(r5)
    print(f"  Rule 5 — Regime Deterioration:  [{r5['severity']}] {r5['message']}")

    r6 = check_market_anomaly(regime_current, regime_history)
    rules.append(r6)
    print(f"  ML    — Market Anomaly (IF):    [{r6['severity']}] {r6['message']}")

    # ── AGGREGATE ─────────────────────────────────────────────────────
    aggregate = compute_aggregate_risk(rules)
    base_equity = portfolio_data.get('equity_allocation', 0)
    adjusted_equity = round(base_equity * aggregate['position_size_mult'], 4)
    aggregate['adjusted_equity_pct'] = round(adjusted_equity * 100, 1)
    aggregate['base_equity_pct']     = round(base_equity * 100, 1)

    print(f"\n{'='*60}")
    print(f"  RISK SUMMARY")
    print(f"{'='*60}")
    print(f"  Risk Score:        {aggregate['risk_score']}/100")
    print(f"  Max Severity:      {aggregate['max_severity']}")
    print(f"  Rules Triggered:   {aggregate['triggered_count']}/6")
    print(f"  Position Mult:     {aggregate['position_size_mult']}x")
    print(f"  Base Equity:       {aggregate['base_equity_pct']}%")
    print(f"  Adjusted Equity:   {aggregate['adjusted_equity_pct']}%")
    print(f"  Message:           {aggregate['overall_message']}")
    print(f"{'='*60}\n")

    # ── SAVE OUTPUT ────────────────────────────────────────────────────
    output = {
        'date':          datetime.today().strftime('%Y-%m-%d'),
        'regime_label':  regime_label,
        'composite_score': comp_score,
        'aggregate':     aggregate,
        'rules':         rules,
        'thresholds':    THRESHOLDS,
    }

    with open(RISK_CURR, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Saved: {RISK_CURR}\n")


if __name__ == "__main__":
    run()
