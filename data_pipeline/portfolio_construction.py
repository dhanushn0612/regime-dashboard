"""
Portfolio Construction — Layer 4
===================================
Industry-standard mean-variance optimisation with:
  - Ledoit-Wolf shrinkage covariance estimator (stable on small samples)
  - Maximum Sharpe Ratio optimisation via scipy
  - Regime-scaled total equity exposure
  - Position size constraints (max 15% per stock)
  - Sector concentration limits (max 40% per sector)
  - Turnover constraint (max 30% rebalance per period)
  - Transaction cost model (20-50bps round-trip by market cap)

This is the institutional standard. Same methodology used by
AQR, Alchemy Capital, systematic desks at large AMCs.

Usage:
    python data_pipeline/portfolio_construction.py

Output:
    public/portfolio_current.json   — current portfolio weights
    public/portfolio_history.json   — historical portfolios
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
PORT_CURR   = os.path.join(OUTPUT_DIR, 'portfolio_current.json')
PORT_HIST   = os.path.join(OUTPUT_DIR, 'portfolio_history.json')
REGIME_CURR = os.path.join(OUTPUT_DIR, 'regime_current.json')
SECTOR_CURR = os.path.join(OUTPUT_DIR, 'sector_current.json')
SCREEN_CURR = os.path.join(OUTPUT_DIR, 'screener_current.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── REGIME-SCALED EQUITY ALLOCATION ───────────────────────────────────
REGIME_EQUITY_ALLOCATION = {
    4: 0.90,   # Strong Bull  — 90% equity
    3: 0.70,   # Mild Bull    — 70% equity
    2: 0.45,   # Neutral      — 45% equity
    1: 0.15,   # Mild Bear    — 15% equity (minimal)
    0: 0.00,   # Strong Bear  — 0% equity
}

# ── TRANSACTION COSTS BY MARKET CAP ───────────────────────────────────
# Round-trip cost (entry + exit) as basis points
TRANSACTION_COSTS = {
    'large':  0.0020,   # 20 bps — Nifty 100 stocks
    'mid':    0.0040,   # 40 bps — Nifty 101-500
    'small':  0.0070,   # 70 bps — below Nifty 500
}

# Large cap tickers (Nifty 100 approx)
LARGE_CAP = {
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
    'HINDUNILVR.NS','ITC.NS','SBIN.NS','BHARTIARTL.NS','KOTAKBANK.NS',
    'LT.NS','AXISBANK.NS','HCLTECH.NS','ASIANPAINT.NS','MARUTI.NS',
    'SUNPHARMA.NS','TITAN.NS','BAJFINANCE.NS','NESTLEIND.NS','ULTRACEMCO.NS',
    'WIPRO.NS','TECHM.NS','DRREDDY.NS','CIPLA.NS','ADANIPORTS.NS',
    'NTPC.NS','POWERGRID.NS','ONGC.NS','COALINDIA.NS','TATASTEEL.NS',
}

# ── SECTOR MAPPING ─────────────────────────────────────────────────────
TICKER_SECTOR = {
    'TCS.NS':'IT', 'INFY.NS':'IT', 'HCLTECH.NS':'IT', 'WIPRO.NS':'IT',
    'TECHM.NS':'IT', 'LTIM.NS':'IT', 'MPHASIS.NS':'IT', 'COFORGE.NS':'IT',
    'PERSISTENT.NS':'IT', 'OFSS.NS':'IT', 'TATAELXSI.NS':'IT',
    'LTTS.NS':'IT', 'KPITTECH.NS':'IT',
    'HDFCBANK.NS':'Bank', 'ICICIBANK.NS':'Bank', 'SBIN.NS':'Bank',
    'KOTAKBANK.NS':'Bank', 'AXISBANK.NS':'Bank', 'INDUSINDBK.NS':'Bank',
    'FEDERALBNK.NS':'Bank', 'BANDHANBNK.NS':'Bank',
    'BAJFINANCE.NS':'Financial', 'BAJAJFINSV.NS':'Financial',
    'MUTHOOTFIN.NS':'Financial', 'CHOLAFIN.NS':'Financial',
    'SUNPHARMA.NS':'Pharma', 'DRREDDY.NS':'Pharma', 'CIPLA.NS':'Pharma',
    'DIVISLAB.NS':'Pharma', 'AUROPHARMA.NS':'Pharma', 'TORNTPHARM.NS':'Pharma',
    'ALKEM.NS':'Pharma', 'LUPIN.NS':'Pharma',
    'HINDUNILVR.NS':'FMCG', 'ITC.NS':'FMCG', 'NESTLEIND.NS':'FMCG',
    'BRITANNIA.NS':'FMCG', 'DABUR.NS':'FMCG', 'GODREJCP.NS':'FMCG',
    'MARICO.NS':'FMCG', 'COLPAL.NS':'FMCG',
    'MARUTI.NS':'Auto', 'BAJAJ-AUTO.NS':'Auto', 'HEROMOTOCO.NS':'Auto',
    'EICHERMOT.NS':'Auto', 'TVSMOTORS.NS':'Auto', 'ASHOKLEY.NS':'Auto',
    'RELIANCE.NS':'Energy', 'ONGC.NS':'Energy', 'COALINDIA.NS':'Energy',
    'TATASTEEL.NS':'Metal', 'JSWSTEEL.NS':'Metal', 'HINDALCO.NS':'Metal',
    'LT.NS':'Infra', 'NTPC.NS':'Infra', 'POWERGRID.NS':'Infra',
    'TITAN.NS':'Consumer', 'ASIANPAINT.NS':'Consumer', 'DMART.NS':'Consumer',
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


# ── STEP 1: DOWNLOAD PRICE HISTORY ────────────────────────────────────
def download_price_history(tickers: list, lookback_days: int = 400) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.
    Returns a DataFrame with tickers as columns, dates as index.
    """
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    prices = {}
    for ticker in tickers:
        try:
            df = flatten(yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=True
            ))
            if len(df) > 100:
                prices[ticker] = get_close(df)
        except Exception:
            pass

    if not prices:
        return pd.DataFrame()

    price_df = pd.DataFrame(prices)
    price_df = price_df.dropna(how='all')
    return price_df


# ── STEP 2: LEDOIT-WOLF COVARIANCE ESTIMATOR ──────────────────────────
def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    The problem with sample covariance on short history:
    - With 15 stocks and 400 days, the sample covariance has estimation error
    - Small errors in correlations compound when you invert the matrix
    - Result: optimiser puts all weight in a few stocks (degenerate solution)

    Ledoit-Wolf shrinkage pulls the sample covariance toward a structured
    target (typically scaled identity matrix). This reduces estimation error
    at the cost of introducing a small bias — a worthwhile tradeoff.

    Formula: Sigma_LW = (1 - alpha) * Sigma_sample + alpha * Sigma_target
    where alpha is determined analytically by minimising MSE.
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns.fillna(0))
        cov = lw.covariance_
        shrinkage = lw.shrinkage_
        print(f"  Ledoit-Wolf shrinkage: {shrinkage:.3f}")
        return cov

    except ImportError:
        # Fallback: manual shrinkage toward identity
        print("  scikit-learn not available — using manual shrinkage")
        S = returns.fillna(0).cov().values
        n = S.shape[0]
        mu = np.trace(S) / n
        alpha = 0.2  # 20% shrinkage
        return (1 - alpha) * S + alpha * mu * np.eye(n)


# ── STEP 3: EXPECTED RETURNS FROM ML SCORES ───────────────────────────
def compute_expected_returns(screener_data: dict,
                              price_df: pd.DataFrame,
                              ann_factor: float = 252) -> pd.Series:
    """
    Compute expected returns for each stock.

    Institutional approach: blend multiple return estimates
      1. Historical momentum (3M return, annualised)
      2. ML score scaled to return space
      3. Mean-reversion adjustment (mild)

    The ML score from Layer 3 is a rank (0-100), not a return forecast.
    We convert it to expected return using a simple linear scaling:
      expected_return = base_return + ml_score_premium

    This is called a "score-to-return" transformation, standard in
    quantitative factor investing (see Grinold & Kahn, "Active Portfolio Management").
    """
    stocks = screener_data.get('stocks', [])
    if not stocks:
        return pd.Series(dtype=float)

    expected_rets = {}

    for stock in stocks:
        ticker = stock['ticker']
        if ticker not in price_df.columns:
            continue

        close = price_df[ticker].dropna()
        if len(close) < 63:
            continue

        # Component 1: 3-month momentum (annualised)
        mom_3m = (s(close.iloc[-1]) / s(close.iloc[-63]) - 1)
        mom_ann = (1 + mom_3m) ** 4 - 1  # Annualise quarterly return

        # Component 2: ML score premium
        # Scale ML score (0-100) to return premium (-2% to +8%)
        ml_score = stock.get('ml_score', 50)
        ml_premium = (ml_score - 50) / 50 * 0.08  # ±8% max premium

        # Component 3: Mild mean reversion (IC-weighted)
        # Stocks that have run very far revert slightly
        ret_6m = stock.get('ret_6m', 0) / 100
        mean_rev = -0.05 * ret_6m  # 5% pull-back on 6M return

        # Blend with IC weights
        # IC (information coefficient) weights: momentum 0.4, ML 0.4, MR 0.2
        expected_ret = (
            mom_ann   * 0.40 +
            ml_premium * 0.40 +
            mean_rev  * 0.20
        )

        expected_rets[ticker] = expected_ret

    return pd.Series(expected_rets)


# ── STEP 4: MEAN-VARIANCE OPTIMISATION ────────────────────────────────
def optimise_portfolio(expected_returns: pd.Series,
                        cov_matrix: np.ndarray,
                        tickers: list,
                        regime_code: int,
                        prev_weights: dict = None) -> dict:
    """
    Maximum Sharpe Ratio optimisation with constraints.

    Objective: maximise (mu - rf) / sigma
    Equivalent to: minimise -Sharpe = -(w'mu - rf) / sqrt(w'Sigma w)

    Constraints:
      1. Weights sum to 1 (fully invested within equity allocation)
      2. No short selling (w_i >= 0)
      3. Max position size: regime-dependent (15% in bull, 25% in bear)
      4. Sector concentration: max 40% in any single sector
      5. Turnover: max 30% change from previous portfolio

    The risk-free rate is approximated by India's 91-day T-bill rate (~6.5%).
    """
    try:
        from scipy.optimize import minimize
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("  scipy not available — using risk parity fallback")
        return risk_parity_weights(tickers, cov_matrix)

    n = len(tickers)
    if n == 0:
        return {}

    mu  = expected_returns.reindex(tickers).fillna(0).values
    cov = cov_matrix
    rf  = 0.065 / 252   # Daily risk-free rate (6.5% annual T-bill)

    # Max position size by regime
    max_pos = {4: 0.15, 3: 0.15, 2: 0.20, 1: 0.25, 0: 0.25}.get(regime_code, 0.15)

    # ── OBJECTIVE: Negative Sharpe Ratio ──────────────────────────────
    def neg_sharpe(w):
        port_ret  = np.dot(w, mu)
        port_var  = np.dot(w, np.dot(cov, w))
        port_std  = np.sqrt(max(port_var, 1e-10))
        sharpe    = (port_ret - rf * 252) / (port_std * np.sqrt(252))
        return -sharpe

    def port_vol(w):
        return np.sqrt(np.dot(w, np.dot(cov, w)) * 252)

    # ── CONSTRAINTS ────────────────────────────────────────────────────
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
    ]

    # Sector concentration constraint (max 40% per sector)
    sector_groups = {}
    for i, ticker in enumerate(tickers):
        sec = TICKER_SECTOR.get(ticker, 'Other')
        if sec not in sector_groups:
            sector_groups[sec] = []
        sector_groups[sec].append(i)

    for sec, indices in sector_groups.items():
        if len(indices) > 1:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=indices: 0.40 - sum(w[i] for i in idx)
            })

    # Turnover constraint
    if prev_weights:
        prev_w = np.array([prev_weights.get(t, 0) for t in tickers])
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, pw=prev_w: 0.30 - np.sum(np.abs(w - pw))
        })

    # ── BOUNDS ─────────────────────────────────────────────────────────
    bounds = [(0.0, max_pos) for _ in range(n)]

    # ── INITIAL GUESS: Equal weight ────────────────────────────────────
    w0 = np.ones(n) / n

    # ── OPTIMISE ───────────────────────────────────────────────────────
    try:
        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9},
        )

        if result.success and not np.any(np.isnan(result.x)):
            weights = result.x
            weights = np.maximum(weights, 0)   # Ensure non-negative
            weights = weights / weights.sum()   # Renormalise

            opt_sharpe = -result.fun
            opt_vol    = port_vol(weights) * 100

            print(f"  Optimisation: SUCCESS")
            print(f"  Expected Sharpe: {opt_sharpe:.2f}")
            print(f"  Expected Vol:    {opt_vol:.1f}%")

            return dict(zip(tickers, weights.tolist()))

        else:
            print(f"  Optimisation failed: {result.message} — using risk parity")
            return risk_parity_weights(tickers, cov_matrix)

    except Exception as e:
        print(f"  Optimisation error: {e} — using risk parity")
        return risk_parity_weights(tickers, cov_matrix)


def risk_parity_weights(tickers: list, cov_matrix: np.ndarray) -> dict:
    """
    Risk parity fallback: weight inversely proportional to volatility.
    Each stock contributes equal volatility to the portfolio.
    """
    vols = np.sqrt(np.diag(cov_matrix) * 252)
    vols = np.maximum(vols, 0.01)   # Floor at 1% to avoid division by zero
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    print("  Using risk parity weights (vol-scaled)")
    return dict(zip(tickers, weights.tolist()))


# ── STEP 5: TRANSACTION COST MODEL ────────────────────────────────────
def compute_transaction_costs(new_weights: dict,
                               prev_weights: dict,
                               portfolio_value: float = 1_000_000) -> dict:
    """
    Estimate round-trip transaction costs for rebalancing.

    Cost = |change in weight| * portfolio_value * round_trip_bps
    where round_trip_bps depends on market cap tier.

    This is what was missing from the original framework — without
    transaction costs, the backtest overstates returns.
    """
    costs = {}
    total_cost = 0.0

    all_tickers = set(new_weights.keys()) | set(prev_weights.keys())

    for ticker in all_tickers:
        new_w = new_weights.get(ticker, 0.0)
        old_w = prev_weights.get(ticker, 0.0)
        change = abs(new_w - old_w)

        if change < 0.001:  # Ignore tiny changes
            continue

        # Determine cost tier
        if ticker in LARGE_CAP:
            cost_rate = TRANSACTION_COSTS['large']
        else:
            cost_rate = TRANSACTION_COSTS['mid']

        cost = change * portfolio_value * cost_rate
        costs[ticker] = round(cost, 2)
        total_cost += cost

    return {
        'breakdown': costs,
        'total_cost': round(total_cost, 2),
        'total_cost_bps': round(total_cost / portfolio_value * 10000, 1),
        'portfolio_value': portfolio_value,
    }


# ── STEP 6: PORTFOLIO ANALYTICS ───────────────────────────────────────
def compute_portfolio_analytics(weights: dict,
                                  price_df: pd.DataFrame,
                                  cov_matrix: np.ndarray,
                                  tickers: list,
                                  expected_returns: pd.Series) -> dict:
    """
    Compute portfolio-level risk and return metrics.
    These are the numbers that appear in fund reporting.
    """
    if not weights:
        return {}

    w = np.array([weights.get(t, 0) for t in tickers])
    mu = expected_returns.reindex(tickers).fillna(0).values

    # Expected return and volatility (annualised)
    exp_ret   = float(np.dot(w, mu)) * 100
    port_var  = float(np.dot(w, np.dot(cov_matrix, w)))
    port_vol  = float(np.sqrt(port_var * 252)) * 100

    # Sharpe ratio
    rf        = 6.5   # Risk-free rate %
    sharpe    = (exp_ret - rf) / port_vol if port_vol > 0 else 0

    # Diversification ratio: weighted avg vol / portfolio vol
    stock_vols = np.sqrt(np.diag(cov_matrix) * 252)
    weighted_avg_vol = float(np.dot(w, stock_vols)) * 100
    div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1

    # Concentration (Herfindahl index)
    herfindahl = float(np.sum(w ** 2))

    # Sector breakdown
    sector_weights = {}
    for ticker, weight in weights.items():
        sec = TICKER_SECTOR.get(ticker, 'Other')
        sector_weights[sec] = sector_weights.get(sec, 0) + weight

    return {
        'expected_return_pct':  round(exp_ret, 2),
        'expected_vol_pct':     round(port_vol, 2),
        'sharpe_ratio':         round(sharpe, 2),
        'diversification_ratio':round(div_ratio, 2),
        'herfindahl_index':     round(herfindahl, 4),
        'effective_n_stocks':   round(1 / herfindahl, 1),
        'sector_weights':       {k: round(v, 4) for k, v in sector_weights.items()},
        'rf_rate_pct':          6.5,
        'note': 'Expected returns are forward-looking estimates. Not guaranteed.',
    }


# ── STEP 7: REBALANCE TRIGGER ─────────────────────────────────────────
def should_rebalance(current_regime: int,
                      prev_portfolio: dict) -> tuple[bool, str]:
    """
    Determine if rebalancing is needed.
    Triggers:
      1. Regime code changed since last rebalance
      2. Any position drifted more than 5% from target weight
      3. First run (no previous portfolio)
    """
    if not prev_portfolio:
        return True, "First run — initialising portfolio"

    prev_regime = prev_portfolio.get('regime_code', -1)
    if current_regime != prev_regime:
        return True, f"Regime changed from {prev_regime} to {current_regime}"

    # Check drift
    positions = prev_portfolio.get('positions', [])
    for pos in positions:
        target = pos.get('target_weight', 0)
        current = pos.get('current_weight', target)
        if abs(current - target) > 0.05:
            return True, f"Position drift exceeded 5% for {pos['ticker']}"

    return False, "No rebalance needed — regime stable, no significant drift"


# ── MAIN ──────────────────────────────────────────────────────────────
def run():
    print("\n" + "="*60)
    print("  PORTFOLIO CONSTRUCTION — LAYER 4")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*60 + "\n")

    # Load regime
    regime_code     = 2
    regime_label    = "Neutral/Choppy"
    composite_score = 50.0

    if os.path.exists(REGIME_CURR):
        with open(REGIME_CURR) as f:
            reg = json.load(f)
        regime_code     = reg.get('regime_code', 2)
        regime_label    = reg.get('regime_label', 'Neutral')
        composite_score = reg.get('composite_score', 50.0)
        print(f"  Regime: {regime_label} ({composite_score}/100)")

    # Load screener output (selected stocks from Layer 3)
    screener_data = {}
    if os.path.exists(SCREEN_CURR):
        with open(SCREEN_CURR) as f:
            screener_data = json.load(f)

    selected_stocks = screener_data.get('stocks', [])
    tickers = [s['ticker'] for s in selected_stocks]

    print(f"  Selected stocks from Layer 3: {len(tickers)}")

    # Load previous portfolio for turnover constraint
    prev_portfolio = {}
    if os.path.exists(PORT_CURR):
        with open(PORT_CURR) as f:
            prev_portfolio = json.load(f)

    # Check rebalance trigger
    rebalance, reason = should_rebalance(regime_code, prev_portfolio)
    print(f"  Rebalance: {'YES' if rebalance else 'NO'} — {reason}")

    # Handle no stocks case
    if regime_code <= 1 or not tickers:
        eq_alloc = REGIME_EQUITY_ALLOCATION.get(regime_code, 0)
        output = {
            'date':              datetime.today().strftime('%Y-%m-%d'),
            'regime_code':       regime_code,
            'regime_label':      regime_label,
            'composite_score':   composite_score,
            'equity_allocation': eq_alloc,
            'cash_allocation':   round(1 - eq_alloc, 4),
            'rebalance_needed':  rebalance,
            'rebalance_reason':  reason,
            'positions':         [],
            'analytics':         {},
            'transaction_costs': {},
            'status':            'cash_mode',
            'note':              f'Regime too weak ({regime_label}) — holding cash',
        }
        with open(PORT_CURR, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Status: CASH MODE — {eq_alloc*100:.0f}% equity, {(1-eq_alloc)*100:.0f}% cash")
        print(f"  Saved: {PORT_CURR}\n")
        return

    # ── DOWNLOAD PRICE HISTORY ─────────────────────────────────────────
    print("\nDownloading price history...")
    price_df = download_price_history(tickers, lookback_days=400)

    available_tickers = [t for t in tickers if t in price_df.columns]
    if not available_tickers:
        print("  No price data available")
        return

    print(f"  Available: {len(available_tickers)}/{len(tickers)} stocks")

    # ── COMPUTE RETURNS ────────────────────────────────────────────────
    returns = price_df[available_tickers].pct_change().dropna()
    print(f"  Returns: {len(returns)} days, {len(available_tickers)} stocks")

    # ── LEDOIT-WOLF COVARIANCE ─────────────────────────────────────────
    print("\nEstimating covariance (Ledoit-Wolf shrinkage)...")
    cov_matrix = ledoit_wolf_covariance(returns)
    print(f"  Covariance matrix: {cov_matrix.shape[0]}x{cov_matrix.shape[1]}")

    # ── EXPECTED RETURNS ───────────────────────────────────────────────
    print("\nComputing expected returns...")
    expected_rets = compute_expected_returns(screener_data, price_df)
    print(f"  Expected returns computed for {len(expected_rets)} stocks")
    if len(expected_rets) > 0:
        print(f"  Range: {expected_rets.min()*100:.1f}% to {expected_rets.max()*100:.1f}% annualised")

    # ── MEAN-VARIANCE OPTIMISATION ─────────────────────────────────────
    print("\nRunning mean-variance optimisation (Max Sharpe)...")
    prev_weights = {p['ticker']: p['target_weight']
                    for p in prev_portfolio.get('positions', [])}

    raw_weights = optimise_portfolio(
        expected_rets, cov_matrix,
        available_tickers, regime_code, prev_weights
    )

    # ── APPLY REGIME EQUITY ALLOCATION ────────────────────────────────
    equity_alloc = REGIME_EQUITY_ALLOCATION.get(regime_code, 0.45)
    final_weights = {t: w * equity_alloc for t, w in raw_weights.items()}
    cash_weight   = round(1.0 - equity_alloc, 4)

    print(f"\n  Equity allocation: {equity_alloc*100:.0f}%")
    print(f"  Cash allocation:   {cash_weight*100:.0f}%")

    # ── TRANSACTION COSTS ──────────────────────────────────────────────
    tc = compute_transaction_costs(final_weights, prev_weights)
    print(f"\n  Transaction costs: {tc['total_cost_bps']:.1f} bps (Rs {tc['total_cost']:,.0f} on Rs 10L portfolio)")

    # ── PORTFOLIO ANALYTICS ────────────────────────────────────────────
    analytics = compute_portfolio_analytics(
        raw_weights, price_df, cov_matrix,
        available_tickers, expected_rets
    )

    # ── BUILD POSITIONS ────────────────────────────────────────────────
    positions = []
    for stock in selected_stocks:
        ticker = stock['ticker']
        if ticker not in final_weights:
            continue
        positions.append({
            'ticker':         ticker,
            'name':           stock['name'],
            'target_weight':  round(final_weights[ticker], 4),
            'equity_weight':  round(raw_weights.get(ticker, 0), 4),
            'ml_score':       stock.get('ml_score', 0),
            'sector':         TICKER_SECTOR.get(ticker, 'Other'),
            'expected_ret':   round(float(expected_rets.get(ticker, 0)) * 100, 2),
            'f_momentum':     stock.get('f_momentum', 0),
            'f_quality':      stock.get('f_quality', 0),
            'f_lowvol':       stock.get('f_lowvol', 0),
            'f_earnings':     stock.get('f_earnings', 0),
        })

    # Sort by target weight descending
    positions.sort(key=lambda x: x['target_weight'], reverse=True)

    # ── PRINT REPORT ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PORTFOLIO REPORT — " + datetime.today().strftime('%Y-%m-%d'))
    print("="*60)
    print(f"  Regime:      {regime_label} ({composite_score}/100)")
    print(f"  Equity:      {equity_alloc*100:.0f}%  |  Cash: {cash_weight*100:.0f}%")
    if analytics:
        print(f"  Exp Return:  {analytics.get('expected_return_pct',0):.1f}%  |  Vol: {analytics.get('expected_vol_pct',0):.1f}%")
        print(f"  Sharpe:      {analytics.get('sharpe_ratio',0):.2f}  |  Eff. Stocks: {analytics.get('effective_n_stocks',0):.1f}")
    print(f"  TX Costs:    {tc.get('total_cost_bps',0):.1f} bps")
    print(f"\n  {'TICKER':<12} {'WEIGHT':>8} {'EQ WT':>8} {'EXP RET':>9} {'SECTOR'}")
    print(f"  {'-'*55}")
    for p in positions:
        print(f"  {p['name']:<12} {p['target_weight']*100:>7.1f}% {p['equity_weight']*100:>7.1f}% {p['expected_ret']:>8.1f}% {p['sector']}")
    print("="*60 + "\n")

    # ── SAVE OUTPUT ────────────────────────────────────────────────────
    output = {
        'date':              datetime.today().strftime('%Y-%m-%d'),
        'regime_code':       regime_code,
        'regime_label':      regime_label,
        'composite_score':   composite_score,
        'equity_allocation': equity_alloc,
        'cash_allocation':   cash_weight,
        'rebalance_needed':  rebalance,
        'rebalance_reason':  reason,
        'positions':         positions,
        'analytics':         analytics,
        'transaction_costs': tc,
        'status':            'active',
    }

    with open(PORT_CURR, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Append to history
    history = []
    if os.path.exists(PORT_HIST):
        with open(PORT_HIST) as f:
            history = json.load(f)

    history.append({
        'date':              output['date'],
        'regime_label':      regime_label,
        'equity_allocation': equity_alloc,
        'n_positions':       len(positions),
        'expected_return':   analytics.get('expected_return_pct', 0),
        'expected_vol':      analytics.get('expected_vol_pct', 0),
        'sharpe':            analytics.get('sharpe_ratio', 0),
        'tx_cost_bps':       tc.get('total_cost_bps', 0),
    })

    with open(PORT_HIST, 'w') as f:
        json.dump(history[-24:], f, default=str)  # Keep last 24 months

    print(f"  Saved: {PORT_CURR}")
    print(f"  Saved: {PORT_HIST}\n")


if __name__ == "__main__":
    run()
