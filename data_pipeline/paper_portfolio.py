"""
Paper Portfolio Tracker
========================
Maintains a virtual portfolio for paper trading the quant strategy.
Tracks positions, cost basis, unrealised P&L, drawdown, factor exposure.
Updated daily. Generates a daily one-page report.

Usage:
    # First time — initialise with starting capital
    python data_pipeline/paper_portfolio.py --init --capital 2500000

    # Daily update (run after regime_monitor.py)
    python data_pipeline/paper_portfolio.py

    # Execute a rebalance (after screener runs)
    python data_pipeline/paper_portfolio.py --rebalance

    # View current state
    python data_pipeline/paper_portfolio.py --report

Output:
    data_pipeline/paper_portfolio.json  — portfolio state (persisted)
    public/paper_portfolio.json         — dashboard-ready summary
    public/paper_report.json            — daily report
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from typing import Optional

warnings.filterwarnings('ignore')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, '..', 'public')
STATE_FILE  = os.path.join(SCRIPT_DIR, 'paper_portfolio.json')
DASH_FILE   = os.path.join(OUTPUT_DIR, 'paper_portfolio.json')
REPORT_FILE = os.path.join(OUTPUT_DIR, 'paper_report.json')
TRADES_FILE = os.path.join(SCRIPT_DIR, 'paper_trades.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def today_str():
    return date.today().strftime('%Y-%m-%d')


# ── LOAD / SAVE STATE ─────────────────────────────────────────────────
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def load_trades() -> list:
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []

def save_trades(trades: list):
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


# ── INITIALISE PORTFOLIO ──────────────────────────────────────────────
def initialise(capital: float) -> dict:
    """
    Create a fresh paper portfolio.
    Called once with --init flag.
    """
    state = {
        'created_at':      today_str(),
        'capital':         capital,
        'cash':            capital,
        'positions':       {},        # ticker -> {shares, cost_price, cost_basis, entry_date}
        'high_water_mark': capital,
        'history': [{
            'date':          today_str(),
            'portfolio_value': capital,
            'cash':          capital,
            'equity_value':  0,
            'equity_alloc':  0,
            'regime_code':   2,
            'regime_score':  50,
            'n_positions':   0,
            'daily_pnl':     0,
            'cumulative_pnl': 0,
            'drawdown':      0,
        }],
        'trades':          [],
        'regime_code':     2,
        'regime_score':    50.0,
        'last_updated':    today_str(),
        'version':         '1.0',
    }
    save_state(state)
    print(f"Portfolio initialised with ₹{capital:,.0f}")
    return state


# ── FETCH CURRENT PRICES ──────────────────────────────────────────────
def fetch_prices(tickers: list) -> dict:
    """Fetch latest prices for all held positions."""
    prices = {}
    if not tickers:
        return prices

    for ticker in tickers:
        try:
            df = flatten(yf.download(ticker, period='5d',
                                     progress=False, auto_adjust=True))
            if len(df) > 0:
                prices[ticker] = round(s(get_close(df).iloc[-1]), 2)
        except Exception:
            pass

    return prices


# ── COMPUTE PORTFOLIO VALUE ───────────────────────────────────────────
def compute_portfolio_value(state: dict, prices: dict) -> dict:
    """
    Compute current portfolio value and metrics.
    Returns a snapshot dict.
    """
    positions = state.get('positions', {})
    cash = state.get('cash', 0)

    equity_value = 0
    position_details = []

    for ticker, pos in positions.items():
        current_price = prices.get(ticker, pos.get('cost_price', 0))
        shares        = pos.get('shares', 0)
        cost_price    = pos.get('cost_price', 0)
        cost_basis    = pos.get('cost_basis', 0)
        market_value  = shares * current_price
        unrealised_pnl = market_value - cost_basis
        unrealised_pct = (unrealised_pnl / cost_basis * 100) if cost_basis > 0 else 0

        equity_value += market_value
        position_details.append({
            'ticker':          ticker,
            'shares':          shares,
            'cost_price':      round(cost_price, 2),
            'current_price':   round(current_price, 2),
            'cost_basis':      round(cost_basis, 2),
            'market_value':    round(market_value, 2),
            'unrealised_pnl':  round(unrealised_pnl, 2),
            'unrealised_pct':  round(unrealised_pct, 2),
            'entry_date':      pos.get('entry_date', ''),
            'weight':          0,  # filled below
        })

    total_value  = cash + equity_value
    equity_alloc = equity_value / total_value if total_value > 0 else 0

    # Fill weights
    for p in position_details:
        p['weight'] = round(p['market_value'] / total_value * 100, 1) if total_value > 0 else 0

    # Sort by weight
    position_details.sort(key=lambda x: x['weight'], reverse=True)

    # Drawdown
    hwm      = state.get('high_water_mark', total_value)
    hwm      = max(hwm, total_value)
    drawdown = (total_value / hwm - 1) * 100 if hwm > 0 else 0

    # P&L vs start
    capital      = state.get('capital', total_value)
    total_pnl    = total_value - capital
    total_pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0

    # Daily P&L
    history = state.get('history', [])
    prev_value = history[-1].get('portfolio_value', capital) if history else capital
    daily_pnl  = total_value - prev_value
    daily_pnl_pct = (daily_pnl / prev_value * 100) if prev_value > 0 else 0

    return {
        'date':            today_str(),
        'portfolio_value': round(total_value, 2),
        'cash':            round(cash, 2),
        'equity_value':    round(equity_value, 2),
        'equity_alloc':    round(equity_alloc * 100, 1),
        'n_positions':     len(positions),
        'positions':       position_details,
        'total_pnl':       round(total_pnl, 2),
        'total_pnl_pct':   round(total_pnl_pct, 2),
        'daily_pnl':       round(daily_pnl, 2),
        'daily_pnl_pct':   round(daily_pnl_pct, 2),
        'drawdown':        round(drawdown, 2),
        'high_water_mark': round(hwm, 2),
        'capital':         capital,
    }


# ── EXECUTE REBALANCE ─────────────────────────────────────────────────
def execute_rebalance(state: dict) -> dict:
    """
    Read the live screener output and rebalance the paper portfolio.
    Buys stocks in the screener output, sells positions not in screener.
    Respects the regime equity allocation.
    """
    # Load screener output
    screener_path = os.path.join(OUTPUT_DIR, 'screener_current.json')
    portfolio_path = os.path.join(OUTPUT_DIR, 'portfolio_current.json')
    regime_path   = os.path.join(OUTPUT_DIR, 'regime_current.json')

    if not os.path.exists(screener_path):
        print("  Screener output not found — run stock_screener.py first")
        return state

    with open(screener_path) as f:
        screener = json.load(f)
    with open(regime_path) as f:
        regime = json.load(f)

    regime_code  = regime.get('regime_code', 2)
    regime_score = regime.get('composite_score', 50.0)

    # Equity allocation from regime
    eq_alloc_map = {4: 0.90, 3: 0.80, 2: 0.60, 1: 0.15, 0: 0.0}
    target_equity_pct = eq_alloc_map.get(regime_code, 0.60)

    # Momentum overlay
    if regime_code == 3:
        # Check 3M Nifty momentum
        try:
            nifty = flatten(yf.download("^NSEI", period="100d",
                                         progress=False, auto_adjust=True))
            nc = get_close(nifty)
            if len(nc) > 63:
                m3 = (s(nc.iloc[-1]) / s(nc.iloc[-63]) - 1) * 100
                if m3 > 8.0:
                    target_equity_pct = 0.88
        except Exception:
            pass

    # Get target tickers from screener
    top_stocks = screener.get('top_stocks', [])
    target_tickers = [s['ticker'] for s in top_stocks[:15]]  # Max 15

    # Current prices
    all_tickers = list(set(list(state.get('positions', {}).keys()) + target_tickers))
    print(f"  Fetching prices for {len(all_tickers)} tickers...")
    prices = fetch_prices(all_tickers)

    total_value = state['cash']
    for ticker, pos in state.get('positions', {}).items():
        total_value += pos.get('shares', 0) * prices.get(ticker, pos.get('cost_price', 0))

    target_equity = total_value * target_equity_pct
    per_stock_allocation = target_equity / len(target_tickers) if target_tickers else 0

    trades = load_trades()
    n_buys = n_sells = 0

    # SELLS — exit positions not in target
    for ticker in list(state.get('positions', {}).keys()):
        if ticker not in target_tickers:
            pos = state['positions'][ticker]
            price = prices.get(ticker, pos.get('cost_price', 0))
            shares = pos.get('shares', 0)
            proceeds = shares * price * (1 - 0.002)  # 20bps transaction cost

            state['cash'] += proceeds
            realised_pnl = proceeds - pos.get('cost_basis', 0)

            trade = {
                'date':         today_str(),
                'type':         'SELL',
                'ticker':       ticker,
                'shares':       shares,
                'price':        round(price, 2),
                'proceeds':     round(proceeds, 2),
                'cost_basis':   round(pos.get('cost_basis', 0), 2),
                'realised_pnl': round(realised_pnl, 2),
                'reason':       'rebalance_exit',
            }
            trades.append(trade)
            del state['positions'][ticker]
            n_sells += 1
            print(f"  SELL {ticker}: {shares} shares @ ₹{price:.0f} | PnL: ₹{realised_pnl:+,.0f}")

    # BUYS — enter positions in target
    for ticker in target_tickers:
        if ticker not in state.get('positions', {}):
            price = prices.get(ticker)
            if not price or price <= 0:
                print(f"  SKIP {ticker}: price not available")
                continue

            cost = per_stock_allocation * (1 + 0.002)  # 20bps transaction cost
            shares = int(per_stock_allocation / price)
            if shares <= 0:
                continue

            actual_cost = shares * price * (1 + 0.002)
            if actual_cost > state['cash']:
                print(f"  SKIP {ticker}: insufficient cash (need ₹{actual_cost:,.0f})")
                continue

            state['cash'] -= actual_cost
            if 'positions' not in state:
                state['positions'] = {}
            state['positions'][ticker] = {
                'shares':     shares,
                'cost_price': round(price, 2),
                'cost_basis': round(actual_cost, 2),
                'entry_date': today_str(),
            }

            trade = {
                'date':    today_str(),
                'type':    'BUY',
                'ticker':  ticker,
                'shares':  shares,
                'price':   round(price, 2),
                'cost':    round(actual_cost, 2),
                'reason':  'rebalance_entry',
            }
            trades.append(trade)
            n_buys += 1
            print(f"  BUY  {ticker}: {shares} shares @ ₹{price:.0f} | Cost: ₹{actual_cost:,.0f}")

    state['regime_code']  = regime_code
    state['regime_score'] = regime_score
    state['last_updated'] = today_str()
    save_trades(trades)

    print(f"\n  Rebalance complete: {n_buys} buys, {n_sells} sells")
    print(f"  Target equity: {target_equity_pct*100:.0f}% | Cash: ₹{state['cash']:,.0f}")
    return state


# ── COMPUTE PERFORMANCE METRICS ───────────────────────────────────────
def compute_performance(history: list, capital: float) -> dict:
    """Compute Sharpe, max drawdown, CAGR from history."""
    if len(history) < 2:
        return {}

    values = [h.get('portfolio_value', capital) for h in history]
    dates  = pd.to_datetime([h.get('date', '2026-01-01') for h in history])

    returns = pd.Series(values).pct_change().dropna()

    # CAGR
    days  = (dates[-1] - dates[0]).days
    years = days / 365.25
    cagr  = ((values[-1] / values[0]) ** (1/years) - 1) * 100 if years > 0 else 0

    # Sharpe (annualised, rf=6.5%)
    rf_daily = 0.065 / 252
    excess   = returns - rf_daily
    sharpe   = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # Max drawdown
    cum     = (1 + returns).cumprod()
    hwm     = cum.cummax()
    dd      = (cum - hwm) / hwm
    max_dd  = float(dd.min()) * 100

    # Win rate (daily)
    win_rate = (returns > 0).mean() * 100

    return {
        'cagr':      round(cagr, 2),
        'sharpe':    round(sharpe, 2),
        'max_dd':    round(max_dd, 2),
        'win_rate':  round(win_rate, 1),
        'days_live': days,
        'total_ret': round((values[-1]/values[0]-1)*100, 2),
    }


# ── GENERATE DAILY REPORT ─────────────────────────────────────────────
def generate_report(state: dict, snapshot: dict) -> dict:
    """Generate a daily one-page report."""
    history  = state.get('history', [])
    trades   = load_trades()
    perf     = compute_performance(history, state.get('capital', 1))

    # Recent trades (last 10)
    recent_trades = trades[-10:] if trades else []

    # Load regime monitor
    monitor_path = os.path.join(OUTPUT_DIR, 'regime_monitor.json')
    monitor = {}
    if os.path.exists(monitor_path):
        with open(monitor_path) as f:
            monitor = json.load(f)

    # Load action required
    action_path = os.path.join(OUTPUT_DIR, 'action_required.json')
    action = None
    if os.path.exists(action_path):
        with open(action_path) as f:
            action_data = json.load(f)
            if action_data.get('action_required'):
                action = action_data

    report = {
        'date':           today_str(),
        'generated_at':   now_str(),

        # Portfolio summary
        'portfolio': {
            'value':       snapshot['portfolio_value'],
            'cash':        snapshot['cash'],
            'equity':      snapshot['equity_value'],
            'equity_pct':  snapshot['equity_alloc'],
            'positions':   snapshot['n_positions'],
            'capital':     state.get('capital', 0),
        },

        # P&L
        'pnl': {
            'today':       snapshot['daily_pnl'],
            'today_pct':   snapshot['daily_pnl_pct'],
            'total':       snapshot['total_pnl'],
            'total_pct':   snapshot['total_pnl_pct'],
            'drawdown':    snapshot['drawdown'],
            'hwm':         snapshot['high_water_mark'],
        },

        # Performance (grows over time)
        'performance': perf,

        # Regime
        'regime': {
            'code':    state.get('regime_code', 2),
            'score':   state.get('regime_score', 50),
            'label':   monitor.get('regime_label', 'Unknown'),
            'change':  monitor.get('score_change', 0),
            'breadth': monitor.get('breadth', 50),
        },

        # Positions
        'positions': snapshot.get('positions', []),

        # Alerts
        'action_required': action is not None,
        'action':          action,

        # Recent trades
        'recent_trades': recent_trades,

        # Days live
        'days_live': perf.get('days_live', 0),
    }

    return report


# ── PRINT DAILY SUMMARY ───────────────────────────────────────────────
def print_summary(snapshot: dict, report: dict, state: dict):
    perf = report.get('performance', {})

    print(f"\n{'='*55}")
    print(f"  PAPER PORTFOLIO — {today_str()}")
    print(f"{'='*55}")
    print(f"  Portfolio value: ₹{snapshot['portfolio_value']:>12,.0f}")
    print(f"  Cash:            ₹{snapshot['cash']:>12,.0f}  ({100 - snapshot['equity_alloc']:.0f}%)")
    print(f"  Equity:          ₹{snapshot['equity_value']:>12,.0f}  ({snapshot['equity_alloc']:.0f}%)")
    print(f"  Positions:       {snapshot['n_positions']} stocks")
    print(f"\n  Today P&L:       ₹{snapshot['daily_pnl']:>+12,.0f}  ({snapshot['daily_pnl_pct']:+.2f}%)")
    print(f"  Total P&L:       ₹{snapshot['total_pnl']:>+12,.0f}  ({snapshot['total_pnl_pct']:+.2f}%)")
    print(f"  Drawdown:        {snapshot['drawdown']:+.2f}%")

    if perf:
        print(f"\n  Performance ({perf.get('days_live',0)} days live):")
        print(f"  CAGR:    {perf.get('cagr',0):+.1f}% p.a.")
        print(f"  Sharpe:  {perf.get('sharpe',0):.2f}")
        print(f"  Max DD:  {perf.get('max_dd',0):.1f}%")

    print(f"\n  Regime: {report['regime'].get('label','?')} "
          f"(score {report['regime'].get('score',50):.0f}, "
          f"change {report['regime'].get('change',0):+.1f})")

    if snapshot.get('positions'):
        print(f"\n  POSITIONS:")
        print(f"  {'TICKER':<14} {'SHARES':>6} {'ENTRY':>8} {'CURR':>8} {'PNL%':>7} {'WEIGHT':>7}")
        print(f"  {'-'*55}")
        for p in snapshot['positions'][:10]:
            print(f"  {p['ticker'].replace('.NS',''):<14} "
                  f"{p['shares']:>6} "
                  f"₹{p['cost_price']:>7,.0f} "
                  f"₹{p['current_price']:>7,.0f} "
                  f"{p['unrealised_pct']:>+6.1f}% "
                  f"{p['weight']:>6.1f}%")

    if report.get('action_required'):
        action = report['action']
        print(f"\n  ⚠️  ACTION REQUIRED — {action.get('worst_severity','')}")
        print(f"  {action.get('primary_message','')}")
        print(f"  Reduce equity: {action.get('current_equity',0):.0f}% → {action.get('target_equity',0):.0f}%")

    print(f"{'='*55}\n")


# ── MAIN ──────────────────────────────────────────────────────────────
def run(init=False, capital=None, rebalance=False, report_only=False):
    print(f"\n{'='*55}")
    print(f"  PAPER PORTFOLIO TRACKER")
    print(f"  {now_str()}")
    print(f"{'='*55}\n")

    # Initialise
    if init:
        if not capital:
            print("Error: --capital required with --init")
            return
        state = initialise(capital)
        print("Run daily with: python data_pipeline/paper_portfolio.py")
        return

    # Load state
    state = load_state()
    if not state:
        print("Portfolio not initialised. Run:")
        print("  python data_pipeline/paper_portfolio.py --init --capital 2500000")
        return

    # Rebalance
    if rebalance:
        print("Executing rebalance...")
        state = execute_rebalance(state)

    # Fetch prices and compute snapshot
    tickers = list(state.get('positions', {}).keys())
    print(f"Fetching prices for {len(tickers)} positions...")
    prices = fetch_prices(tickers)

    snapshot = compute_portfolio_value(state, prices)

    # Update high water mark
    if snapshot['portfolio_value'] > state.get('high_water_mark', 0):
        state['high_water_mark'] = snapshot['portfolio_value']

    # Update history
    history_entry = {
        'date':             today_str(),
        'portfolio_value':  snapshot['portfolio_value'],
        'cash':             snapshot['cash'],
        'equity_value':     snapshot['equity_value'],
        'equity_alloc':     snapshot['equity_alloc'],
        'regime_code':      state.get('regime_code', 2),
        'regime_score':     state.get('regime_score', 50),
        'n_positions':      snapshot['n_positions'],
        'daily_pnl':        snapshot['daily_pnl'],
        'daily_pnl_pct':    snapshot['daily_pnl_pct'],
        'total_pnl':        snapshot['total_pnl'],
        'total_pnl_pct':    snapshot['total_pnl_pct'],
        'drawdown':         snapshot['drawdown'],
    }

    history = state.get('history', [])
    # Avoid duplicate dates
    if history and history[-1].get('date') == today_str():
        history[-1] = history_entry
    else:
        history.append(history_entry)
    state['history'] = history
    state['last_updated'] = today_str()

    # Generate report
    report = generate_report(state, snapshot)

    # Print summary
    print_summary(snapshot, report, state)

    # Save state and outputs
    save_state(state)

    # Dashboard output (no full history — too large)
    dash = {
        'date':         today_str(),
        'portfolio':    report['portfolio'],
        'pnl':          report['pnl'],
        'performance':  report['performance'],
        'regime':       report['regime'],
        'positions':    snapshot['positions'],
        'action_required': report['action_required'],
        'days_live':    report['days_live'],
        'recent_trades': report['recent_trades'][-5:],
        'history_30d':  history[-30:],
    }
    with open(DASH_FILE, 'w') as f:
        json.dump(dash, f, indent=2, default=str)

    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Saved: {DASH_FILE}")
    print(f"  Saved: {REPORT_FILE}")


if __name__ == "__main__":
    init       = '--init'      in sys.argv
    rebalance  = '--rebalance' in sys.argv
    report_only= '--report'    in sys.argv

    capital = None
    if '--capital' in sys.argv:
        idx = sys.argv.index('--capital')
        if idx + 1 < len(sys.argv):
            try:
                capital = float(sys.argv[idx + 1].replace(',', ''))
            except ValueError:
                print("Invalid capital amount")
                sys.exit(1)

    run(init=init, capital=capital, rebalance=rebalance, report_only=report_only)
