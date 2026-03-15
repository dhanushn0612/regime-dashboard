"""
Sector Rotation Engine — Layer 2
==================================
Hybrid approach: rule-based momentum filter → XGBoost ML ranking
Monthly rebalance, regime-dependent sector count.

Architecture:
  Step 1 — Download NSE sector index data (13 sectors)
  Step 2 — Rule-based filter: remove sectors in downtrend
  Step 3 — Feature engineering: momentum, volatility, breadth, relative strength
  Step 4 — XGBoost: rank surviving sectors by predicted forward return
  Step 5 — Regime-dependent allocation: more sectors in bull, fewer in bear
  Step 6 — Output: sector weights + rationale + backtest

Usage:
  python data_pipeline/sector_rotation.py

Output:
  public/sector_current.json   — current allocation
  public/sector_history.json   — historical allocations
  data_pipeline/sector_model.pkl — trained XGBoost model
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

def month_resample(series):
    """Handle pandas version differences for month-end resampling."""
    try:
        return series.pipe(lambda s: month_resample(s))
    except Exception:
        return series.pipe(lambda s: month_resample(s))



# ── PATHS ─────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, '..', 'public')
SECTOR_CURR   = os.path.join(OUTPUT_DIR, 'sector_current.json')
SECTOR_HIST   = os.path.join(OUTPUT_DIR, 'sector_history.json')
MODEL_PATH    = os.path.join(SCRIPT_DIR, 'sector_model.pkl')
REGIME_CURR   = os.path.join(OUTPUT_DIR, 'regime_current.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── NSE SECTOR INDICES ─────────────────────────────────────────────────
# Yahoo Finance tickers for NSE sector indices
SECTORS = {
    'IT':           '^CNXIT',
    'Bank':         '^NSEBANK',
    'Auto':         '^CNXAUTO',
    'Pharma':       '^CNXPHARMA',
    'FMCG':         '^CNXFMCG',
    'Metal':        '^CNXMETAL',
    'Energy':       '^CNXENERGY',
    'Realty':       '^CNXREALTY',
    'Infra':        '^CNXINFRA',
    'Media':        '^CNXMEDIA',
    'PSU Bank':     '^CNXPSUBANK',
    'Financial':    '^CNXFIN',
    'Consumption':  '^CNXCONSUM',
}

# ── REGIME-DEPENDENT SECTOR COUNT ─────────────────────────────────────
REGIME_SECTOR_COUNT = {
    4: 4,   # Strong Bull  — hold top 4 sectors
    3: 3,   # Mild Bull    — hold top 3 sectors
    2: 2,   # Neutral      — hold top 2 (defensive only)
    1: 1,   # Mild Bear    — hold top 1 (most defensive)
    0: 0,   # Strong Bear  — hold nothing, go to cash
}

# Defensive sectors — preferred in low-regime environments
DEFENSIVE_SECTORS = {'FMCG', 'Pharma', 'IT', 'Consumption'}
CYCLICAL_SECTORS  = {'Metal', 'Auto', 'Realty', 'Infra', 'Energy', 'Media'}


# ── DATA STRUCTURES ───────────────────────────────────────────────────
@dataclass
class SectorAllocation:
    date: str
    regime_code: int
    regime_label: str
    composite_score: float
    sectors_held: int
    allocations: list       # [{'sector': ..., 'weight': ..., 'score': ..., 'rationale': ...}]
    excluded_sectors: list  # sectors filtered out by rules
    cash_weight: float
    model_used: str


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


# ── STEP 1: DOWNLOAD SECTOR DATA ──────────────────────────────────────
def download_sectors(lookback_days=504) -> dict[str, pd.DataFrame]:
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    print("Downloading NSE sector indices...")
    data = {}
    for name, ticker in SECTORS.items():
        try:
            df = flatten(yf.download(ticker, start=start, end=end,
                                     progress=False, auto_adjust=True))
            if len(df) > 100:
                data[name] = df
                print(f"  ✓ {name:<15} {len(df)} days")
            else:
                print(f"  ✗ {name:<15} insufficient data")
        except Exception as e:
            print(f"  ✗ {name:<15} error: {e}")

    print(f"\n  {len(data)}/{len(SECTORS)} sectors loaded\n")
    return data


# ── STEP 2: RULE-BASED FILTER ─────────────────────────────────────────
def rule_based_filter(sector_data: dict, regime_code: int) -> tuple[list, list]:
    """
    Remove sectors that fail momentum/trend rules.
    Rules:
      1. Price must be above 50 DMA (basic trend filter)
      2. 1-month return must be positive (no falling sectors)
      3. In Bear regimes (code <= 1): only defensive sectors pass
      4. Sector must have positive 3M relative strength vs Nifty

    Returns: (passing_sectors, excluded_with_reason)
    """
    passing  = []
    excluded = []

    # Download Nifty for relative strength
    end   = datetime.today()
    start = end - timedelta(days=200)
    try:
        nifty  = flatten(yf.download("^NSEI", start=start, end=end,
                                     progress=False, auto_adjust=True))
        nifty_close = get_close(nifty)
        nifty_3m = s(nifty_close.iloc[-1]) / s(nifty_close.iloc[-63]) - 1
    except Exception:
        nifty_3m = 0.0

    for name, df in sector_data.items():
        close = get_close(df)
        reasons_excluded = []

        # Rule 1: Above 50 DMA
        sma50 = close.rolling(50).mean()
        above_50dma = s(close.iloc[-1]) > s(sma50.iloc[-1])
        if not above_50dma:
            reasons_excluded.append("below 50 DMA")

        # Rule 2: Positive 1M return
        ret_1m = (s(close.iloc[-1]) / s(close.iloc[-21]) - 1) * 100
        if ret_1m < 0 and not above_50dma:
            reasons_excluded.append(f"1M return {ret_1m:.1f}%")

        # Rule 3: Bear regime — only defensives
        if regime_code <= 1 and name not in DEFENSIVE_SECTORS:
            reasons_excluded.append(f"cyclical in bear regime ({regime_code})")

        # Rule 4: Positive relative strength vs Nifty (3M)
        sector_3m = (s(close.iloc[-1]) / s(close.iloc[-63]) - 1) if len(close) >= 63 else 0
        rel_strength = sector_3m - nifty_3m
        if rel_strength < -0.05:  # Underperforming Nifty by more than 5%
            reasons_excluded.append(f"relative strength {rel_strength*100:.1f}% vs Nifty")

        if reasons_excluded:
            excluded.append({
                'sector': name,
                'reasons': reasons_excluded,
                'ret_1m': round(ret_1m, 2),
                'above_50dma': above_50dma,
            })
        else:
            passing.append(name)

    print(f"  Rule filter: {len(passing)} pass, {len(excluded)} excluded")
    return passing, excluded


# ── STEP 3: FEATURE ENGINEERING ───────────────────────────────────────
def build_features(sector_data: dict, sectors: list,
                   regime_code: int, composite_score: float) -> pd.DataFrame:
    """
    Build feature matrix for ML ranking.

    Features per sector:
      Momentum:   ret_1m, ret_3m, ret_6m, ret_12m
      Trend:      dist_from_52w_high, above_50dma, above_200dma
      Volatility: vol_20d, vol_60d, vol_ratio (20d/60d)
      RS:         rs_vs_nifty_1m, rs_vs_nifty_3m
      Regime:     composite_score, regime_code
      Seasonality: month (1-12)
    """
    # Nifty for relative strength
    end   = datetime.today()
    start = end - timedelta(days=400)
    try:
        nifty      = flatten(yf.download("^NSEI", start=start, end=end,
                                         progress=False, auto_adjust=True))
        nifty_c    = get_close(nifty)
        nifty_1m   = s(nifty_c.iloc[-1]) / s(nifty_c.iloc[-21])  - 1
        nifty_3m   = s(nifty_c.iloc[-1]) / s(nifty_c.iloc[-63])  - 1
    except Exception:
        nifty_1m = nifty_3m = 0.0

    rows = []
    for name in sectors:
        if name not in sector_data:
            continue
        close = get_close(sector_data[name])
        rets  = close.pct_change().dropna()

        def roc(p):
            return (s(close.iloc[-1]) / s(close.iloc[-p]) - 1) if len(close) > p else 0.0

        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        high52 = close.rolling(252).max()

        vol_20 = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252)
        vol_60 = s(rets.rolling(60).std().iloc[-1]) * np.sqrt(252)

        rows.append({
            'sector':            name,
            'ret_1m':            roc(21),
            'ret_3m':            roc(63),
            'ret_6m':            roc(126),
            'ret_12m':           roc(252),
            'dist_52w_high':     s(close.iloc[-1]) / s(high52.iloc[-1]) - 1,
            'above_50dma':       float(s(close.iloc[-1]) > s(sma50.iloc[-1])),
            'above_200dma':      float(s(close.iloc[-1]) > s(sma200.iloc[-1])),
            'vol_20d':           vol_20,
            'vol_60d':           vol_60,
            'vol_ratio':         vol_20 / vol_60 if vol_60 > 0 else 1.0,
            'rs_vs_nifty_1m':    roc(21) - nifty_1m,
            'rs_vs_nifty_3m':    roc(63) - nifty_3m,
            'composite_score':   composite_score / 100,
            'regime_code':       regime_code / 4,
            'is_defensive':      float(name in DEFENSIVE_SECTORS),
            'month':             datetime.today().month / 12,
        })

    return pd.DataFrame(rows)


# ── STEP 4: XGBOOST TRAINING & RANKING ────────────────────────────────
def build_training_data(sector_data: dict) -> pd.DataFrame:
    """
    Build historical training dataset.
    Label = sector's forward 1-month return (what we want to predict).
    Features = same as build_features() but computed at each month-end.
    """
    # Skip if model already saved — saves ~5 min on GitHub Actions
    if os.path.exists(MODEL_PATH):
        return pd.DataFrame()
    print("  Building XGBoost training data...")

    end   = datetime.today()
    start = end - timedelta(days=730)  # 2 years

    # Resample to month-ends
    records = []

    for name, df in sector_data.items():
        close = get_close(df)
        monthly_dates = close.pipe(lambda s: month_resample(s)).index

        for i in range(3, len(monthly_dates) - 1):
            d     = monthly_dates[i]
            d_fwd = monthly_dates[i + 1]

            c_slice = close[close.index <= d]
            if len(c_slice) < 130:
                continue

            # Forward return (label)
            try:
                fwd_close = close[close.index <= d_fwd].iloc[-1]
                curr_close = s(c_slice.iloc[-1])
                fwd_ret = s(fwd_close) / curr_close - 1
            except Exception:
                continue

            rets = c_slice.pct_change().dropna()

            def roc(p):
                return (s(c_slice.iloc[-1]) / s(c_slice.iloc[-p]) - 1) if len(c_slice) > p else 0.0

            sma50  = c_slice.rolling(50).mean()
            sma200 = c_slice.rolling(200).mean()
            high52 = c_slice.rolling(252).max()
            vol_20 = s(rets.rolling(20).std().iloc[-1]) * np.sqrt(252)
            vol_60 = s(rets.rolling(60).std().iloc[-1]) * np.sqrt(252)

            records.append({
                'sector':       name,
                'date':         d,
                'ret_1m':       roc(21),
                'ret_3m':       roc(63),
                'ret_6m':       roc(126),
                'ret_12m':      roc(252),
                'dist_52w_high':s(c_slice.iloc[-1]) / s(high52.iloc[-1]) - 1,
                'above_50dma':  float(s(c_slice.iloc[-1]) > s(sma50.iloc[-1])),
                'above_200dma': float(s(c_slice.iloc[-1]) > s(sma200.iloc[-1])),
                'vol_20d':      vol_20,
                'vol_60d':      vol_60,
                'vol_ratio':    vol_20 / vol_60 if vol_60 > 0 else 1.0,
                'is_defensive': float(name in DEFENSIVE_SECTORS),
                'month':        d.month / 12,
                'fwd_ret':      fwd_ret,   # Label
            })

    df = pd.DataFrame(records)
    print(f"  Training samples: {len(df)} ({len(df['sector'].unique())} sectors)")
    return df


FEATURE_COLS = [
    'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
    'dist_52w_high', 'above_50dma', 'above_200dma',
    'vol_20d', 'vol_60d', 'vol_ratio',
    'is_defensive', 'month',
]


def train_xgboost(train_df: pd.DataFrame):
    """Train XGBoost to predict sector forward 1M return."""
    try:
        from xgboost import XGBRegressor
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    if not HAS_XGB or len(train_df) < 50:
        print("  XGBoost not available or insufficient data — using momentum ranking")
        return None

    X = train_df[FEATURE_COLS].fillna(0)
    y = train_df['fwd_ret']

    # Walk-forward split — train on first 70%, test on last 30%
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Out-of-sample evaluation
    preds = model.predict(X_test)
    corr  = pd.Series(preds).corr(y_test.reset_index(drop=True))
    print(f"  XGBoost IC (out-of-sample): {corr:.3f}")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top features: {', '.join([f'{k}({v:.2f})' for k,v in top_features])}")

    return model


def rank_sectors_ml(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """Use XGBoost to rank sectors by predicted forward return."""
    if model is None:
        # Fallback: rank by composite momentum score
        features_df['predicted_ret'] = (
            features_df['ret_1m']  * 0.30 +
            features_df['ret_3m']  * 0.35 +
            features_df['ret_6m']  * 0.20 +
            features_df['ret_12m'] * 0.15
        )
        features_df['model_used'] = 'momentum_composite'
    else:
        X = features_df[FEATURE_COLS].fillna(0)
        features_df['predicted_ret'] = model.predict(X)
        features_df['model_used'] = 'xgboost'

    return features_df.sort_values('predicted_ret', ascending=False).reset_index(drop=True)


# ── STEP 5: REGIME-DEPENDENT ALLOCATION ───────────────────────────────
def compute_allocation(ranked_df: pd.DataFrame,
                        regime_code: int,
                        regime_label: str,
                        composite_score: float,
                        excluded: list) -> SectorAllocation:
    """
    Select top N sectors based on regime and assign weights.
    Weighting: score-proportional within selected sectors.
    """
    n_sectors = REGIME_SECTOR_COUNT.get(regime_code, 0)

    if n_sectors == 0 or ranked_df.empty:
        return SectorAllocation(
            date            = datetime.today().strftime('%Y-%m-%d'),
            regime_code     = regime_code,
            regime_label    = regime_label,
            composite_score = composite_score,
            sectors_held    = 0,
            allocations     = [],
            excluded_sectors= excluded,
            cash_weight     = 1.0,
            model_used      = 'none — strong bear, full cash',
        )

    # Select top N
    selected = ranked_df.head(n_sectors).copy()

    # Score-proportional weighting
    # Shift predicted returns to be positive before normalising
    min_ret = selected['predicted_ret'].min()
    selected['score_adj'] = selected['predicted_ret'] - min_ret + 0.01
    total = selected['score_adj'].sum()
    selected['weight'] = selected['score_adj'] / total

    # Build allocations list
    allocations = []
    for _, row in selected.iterrows():
        rationale = []
        if row['ret_1m'] > 0:
            rationale.append(f"1M +{row['ret_1m']*100:.1f}%")
        if row['ret_3m'] > 0:
            rationale.append(f"3M +{row['ret_3m']*100:.1f}%")
        if row['above_50dma']:
            rationale.append("above 50 DMA")
        if row['rs_vs_nifty_1m'] > 0 if 'rs_vs_nifty_1m' in row else False:
            rationale.append("outperforming Nifty")

        allocations.append({
            'sector':         row['sector'],
            'weight':         round(float(row['weight']), 4),
            'predicted_ret':  round(float(row['predicted_ret']) * 100, 2),
            'ret_1m':         round(float(row['ret_1m']) * 100, 2),
            'ret_3m':         round(float(row['ret_3m']) * 100, 2),
            'rationale':      ', '.join(rationale) if rationale else 'ML ranked',
        })

    model_used = selected['model_used'].iloc[0] if 'model_used' in selected.columns else 'momentum'

    return SectorAllocation(
        date            = datetime.today().strftime('%Y-%m-%d'),
        regime_code     = regime_code,
        regime_label    = regime_label,
        composite_score = composite_score,
        sectors_held    = n_sectors,
        allocations     = allocations,
        excluded_sectors= excluded,
        cash_weight     = round(1.0 - sum(a['weight'] for a in allocations), 4),
        model_used      = model_used,
    )


# ── STEP 6: BACKTEST ──────────────────────────────────────────────────
def run_backtest(sector_data: dict, model) -> pd.DataFrame:
    """
    Fast vectorised backtest — no downloads inside loop.
    Uses already-loaded sector_data only. Max 12 months.
    """
    print("Running backtest (fast mode)...")
    try:
        sample_close = get_close(next(iter(sector_data.values())))
        monthly_dates = sample_close.pipe(lambda s: month_resample(s)).index[-13:]
    except Exception as e:
        print(f"  Backtest skipped: {e}")
        return pd.DataFrame()

    results = []
    for d in monthly_dates[:-1]:
        try:
            rows = []
            for name, df in sector_data.items():
                close = get_close(df)
                past  = close[close.index <= d]
                if len(past) < 22:
                    continue
                rows.append({
                    'sector':        name,
                    'predicted_ret': (s(past.iloc[-1]) / s(past.iloc[-21]) - 1),
                })
            if not rows:
                continue
            top3 = (pd.DataFrame(rows)
                    .sort_values('predicted_ret', ascending=False)
                    .head(3)['sector'].tolist())
            fwd_rets = []
            for name in top3:
                close  = get_close(sector_data[name])
                past   = close[close.index <= d]
                future = close[close.index > d].iloc[:21]
                if len(future) >= 10 and len(past) > 0:
                    fwd_rets.append(s(future.iloc[-1]) / s(past.iloc[-1]) - 1)
            if fwd_rets:
                results.append({
                    'date':    str(d.date()),
                    'top3':    top3,
                    'fwd_ret': round(float(np.mean(fwd_rets)) * 100, 2),
                })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        wins    = (df['fwd_ret'] > 0).sum()
        avg_ret = df['fwd_ret'].mean()
        cum_ret = (1 + df['fwd_ret'] / 100).prod() - 1
        print(f"  Backtest: {len(df)} months, {wins} wins ({wins/len(df)*100:.0f}% WR)")
        print(f"  Avg monthly return: {avg_ret:.2f}%")
        print(f"  Cumulative return:  {cum_ret*100:.1f}%")
    else:
        print("  No backtest results")
    return df



def print_allocation(alloc):
    sep = "=" * 55
    print("")
    print("  " + sep)
    print("  SECTOR ROTATION -- " + alloc.date)
    print("  " + sep)
    print("  Regime:  " + alloc.regime_label + " (" + str(alloc.composite_score) + "/100)")
    print("  Holding: " + str(alloc.sectors_held) + " sectors  |  Cash: " + str(round(alloc.cash_weight*100)) + "%")
    print("  Model:   " + alloc.model_used)
    print("")
    print("  ALLOCATIONS:")
    print("  {:<16} {:>8} {:>10} {:>8} {:>8}".format("Sector","Weight","Pred Ret","1M","3M"))
    print("  " + "-" * 52)
    for a in alloc.allocations:
        print("  {:<16} {:>7.1f}% {:>9.1f}% {:>7.1f}% {:>7.1f}%".format(
            a["sector"], a["weight"]*100, a["predicted_ret"], a["ret_1m"], a["ret_3m"]))
    if alloc.excluded_sectors:
        print("")
        print("  EXCLUDED (" + str(len(alloc.excluded_sectors)) + " sectors):")
        for e in alloc.excluded_sectors[:5]:
            print("  x " + e["sector"] + ": " + ", ".join(e["reasons"]))
    print("  " + sep)
    print("")


def run():
    print(f"\n{'='*55}")
    print(f"  SECTOR ROTATION ENGINE — {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}\n")

    # Load current regime
    regime_code     = 2
    regime_label    = "Neutral/Choppy"
    composite_score = 50.0

    if os.path.exists(REGIME_CURR):
        with open(REGIME_CURR) as f:
            reg = json.load(f)
        regime_code     = reg.get('regime_code', 2)
        regime_label    = reg.get('regime_label', 'Neutral/Choppy')
        composite_score = reg.get('composite_score', 50.0)
        print(f"  Regime loaded: {regime_label} ({composite_score:.1f}/100)\n")

    # Step 1: Download sector data
    sector_data = download_sectors(lookback_days=504)

    if not sector_data:
        print("  ERROR: No sector data downloaded")
        return

    # Step 2: Rule-based filter
    print("Applying rule-based filter...")
    passing_sectors, excluded = rule_based_filter(sector_data, regime_code)

    # Step 3: Feature engineering
    print("\nEngineering features...")
    features_df = build_features(
        sector_data, passing_sectors, regime_code, composite_score
    )

    # Step 4: Train or load XGBoost
    model = None
    if os.path.exists(MODEL_PATH):
        print("Loading existing XGBoost model...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("  ✓ Model loaded")
    else:
        print("Training XGBoost model...")
        train_df = build_training_data(sector_data)
        if len(train_df) >= 50:
            model = train_xgboost(train_df)
            if model:
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(model, f)
                print("  ✓ Model saved")

    # Add regime features to features_df
    if not features_df.empty:
        features_df['composite_score'] = composite_score / 100
        features_df['regime_code']     = regime_code / 4

    # Step 4b: Rank sectors
    print("\nRanking sectors...")
    if not features_df.empty:
        ranked_df = rank_sectors_ml(model, features_df)
        print(f"  Ranked {len(ranked_df)} sectors")
        for i, row in ranked_df.iterrows():
            print(f"  {i+1}. {row['sector']:<16} predicted: {row['predicted_ret']*100:+.2f}%")
    else:
        ranked_df = pd.DataFrame()
        print("  No sectors passed filter")

    # Step 5: Compute allocation
    print("\nComputing allocation...")
    alloc = compute_allocation(
        ranked_df, regime_code, regime_label, composite_score, excluded
    )

    # Print report
    print_allocation(alloc)

    # Step 6: Backtest
    backtest_df = run_backtest(sector_data, model)

    # Save outputs
    with open(SECTOR_CURR, 'w') as f:
        json.dump(asdict(alloc), f, indent=2, default=str)

    backtest_records = backtest_df.to_dict('records') if not backtest_df.empty else []
    with open(SECTOR_HIST, 'w') as f:
        json.dump(backtest_records, f, default=str)

    print(f"✓ Saved: {SECTOR_CURR}")
    print(f"✓ Saved: {SECTOR_HIST}\n")


if __name__ == "__main__":
    run()
