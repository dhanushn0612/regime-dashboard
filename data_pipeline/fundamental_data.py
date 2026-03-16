"""
Fundamental Data Fetcher — Quality & Earnings Upgrade
========================================================
Fetches real fundamental data for NSE stocks using TradingView.
Replaces price-based quality/earnings proxies with actual financials.

Data fetched per stock:
  Quality:  ROE, ROCE, Debt/Equity, Operating Margin, Current Ratio
  Earnings: Revenue Growth (YoY), EPS Growth (YoY), Net Margin trend

Usage:
    python data_pipeline/fundamental_data.py

Output:
    data_pipeline/fundamental_cache.csv  — cached fundamentals
    (auto-detected and used by stock_screener.py)

Install requirement:
    pip install tradingview-scraper
"""

import os
import time
import json
import random
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_CSV    = os.path.join(SCRIPT_DIR, 'fundamental_cache.csv')
CACHE_DAYS   = 7   # Refresh fundamentals weekly (quarterly data doesn't change daily)


# ── NIFTY 500 TICKER MAPPING ──────────────────────────────────────────
# TradingView uses NSE:SYMBOL format (no .NS suffix)
def to_tv_symbol(yf_ticker: str) -> str:
    """Convert yfinance ticker (RELIANCE.NS) to TradingView format (NSE:RELIANCE)"""
    symbol = yf_ticker.replace('.NS', '').replace('.BO', '')
    return f"NSE:{symbol}"


# ── FETCH FUNDAMENTALS FROM TRADINGVIEW ───────────────────────────────
def fetch_tv_fundamentals(tv_symbol: str) -> Optional[dict]:
    """
    Fetch fundamental data for a single stock from TradingView.
    Returns dict with quality and earnings metrics or None if failed.
    """
    try:
        from tradingview_scraper.symbols.overview import Overview
        ov = Overview()

        financials = ov.get_financials(symbol=tv_symbol)
        if not financials or financials.get('status') != 'success':
            return None

        data = financials.get('data', {})

        return {
            # Quality metrics
            'roe':              data.get('return_on_equity_fq'),
            'roa':              data.get('return_on_assets_fq'),
            'debt_to_equity':   data.get('debt_to_equity_fq'),
            'current_ratio':    data.get('current_ratio_fq'),
            'operating_margin': data.get('operating_margin_ttm'),
            'net_margin':       data.get('net_margin_percent_ttm'),
            'gross_margin':     data.get('gross_margin_percent_ttm'),

            # Earnings/growth metrics
            'revenue_ttm':      data.get('total_revenue'),
            'net_income':       data.get('net_income'),
            'free_cash_flow':   data.get('free_cash_flow'),
            'pe_ratio':         data.get('price_earnings_ttm'),
            'market_cap':       data.get('market_cap_basic'),
        }

    except ImportError:
        return None
    except Exception:
        return None


def fetch_tv_performance(tv_symbol: str) -> Optional[dict]:
    """Fetch price performance data from TradingView."""
    try:
        from tradingview_scraper.symbols.overview import Overview
        ov = Overview()

        perf = ov.get_performance(symbol=tv_symbol)
        if not perf or perf.get('status') != 'success':
            return None

        data = perf.get('data', {})
        return {
            'perf_1w':  data.get('Perf.W'),
            'perf_1m':  data.get('Perf.1M'),
            'perf_3m':  data.get('Perf.3M'),
            'perf_6m':  data.get('Perf.6M'),
            'perf_ytd': data.get('Perf.YTD'),
            'perf_1y':  data.get('Perf.Y'),
        }
    except Exception:
        return None


# ── FALLBACK: SCRAPE SCREENER.IN ──────────────────────────────────────
def fetch_screener_fundamentals(nse_symbol: str) -> Optional[dict]:
    """
    Fallback: fetch from screener.in public page.
    No login required for basic metrics.
    Format: https://www.screener.in/company/RELIANCE/consolidated/
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://www.screener.in/company/{nse_symbol}/consolidated/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            # Try standalone page
            url = f"https://www.screener.in/company/{nse_symbol}/"
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                return None

        soup = BeautifulSoup(r.text, 'html.parser')

        result = {}

        # Parse the key ratios section
        # Screener.in shows ratios in a specific div structure
        ratios_section = soup.find('section', id='top-ratios')
        if ratios_section:
            for li in ratios_section.find_all('li'):
                name_tag  = li.find('span', class_='name')
                value_tag = li.find('span', class_='value') or li.find('span', class_='number')
                if name_tag and value_tag:
                    name  = name_tag.get_text(strip=True).lower()
                    value = value_tag.get_text(strip=True).replace(',', '').replace('%', '')
                    try:
                        value = float(value)
                    except Exception:
                        continue

                    if 'roe' in name:
                        result['roe'] = value / 100  # Convert % to decimal
                    elif 'roce' in name:
                        result['roce'] = value / 100
                    elif 'debt' in name and 'equity' in name:
                        result['debt_to_equity'] = value
                    elif 'current ratio' in name:
                        result['current_ratio'] = value
                    elif 'opm' in name or 'operating' in name:
                        result['operating_margin'] = value / 100
                    elif 'npm' in name or 'net profit' in name:
                        result['net_margin'] = value / 100

        # Parse sales growth from quarterly data table
        tables = soup.find_all('table', class_='data-table')
        for table in tables:
            header = table.find('th')
            if header and 'sales' in header.get_text(strip=True).lower():
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if cells and 'sales' in cells[0].get_text(strip=True).lower():
                        try:
                            values = [float(c.get_text(strip=True).replace(',','')) for c in cells[1:] if c.get_text(strip=True)]
                            if len(values) >= 2:
                                result['revenue_growth_yoy'] = (values[-1] / values[-5] - 1) if len(values) >= 5 else 0
                        except Exception:
                            pass

        return result if result else None

    except ImportError:
        return None
    except Exception:
        return None


# ── MAIN FETCHER ──────────────────────────────────────────────────────
def fetch_fundamentals_batch(tickers: list, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch fundamentals for all tickers.
    Uses cache if fresh (< 7 days old).
    Tries TradingView first, falls back to screener.in.
    """
    # Check cache
    if not force_refresh and os.path.exists(CACHE_CSV):
        cache = pd.read_csv(CACHE_CSV, parse_dates=['fetched_at'])
        cache_age = (datetime.now() - cache['fetched_at'].max()).days
        if cache_age < CACHE_DAYS:
            print(f"  Using cached fundamentals ({cache_age} days old, {len(cache)} stocks)")
            return cache

    print(f"  Fetching fundamentals for {len(tickers)} stocks...")

    # Check if tradingview-scraper is available
    try:
        import tradingview_scraper
        HAS_TV = True
        print("  Source: TradingView")
    except ImportError:
        HAS_TV = False
        print("  TradingView scraper not installed — using screener.in fallback")
        print("  Install with: pip install tradingview-scraper")

    records = []
    success = 0
    failed  = 0

    for ticker in tickers:
        nse_symbol = ticker.replace('.NS', '').replace('.BO', '')
        tv_symbol  = f"NSE:{nse_symbol}"

        data = None

        # Try TradingView
        if HAS_TV:
            data = fetch_tv_fundamentals(tv_symbol)
            if data:
                perf = fetch_tv_performance(tv_symbol)
                if perf:
                    data.update(perf)
            time.sleep(random.uniform(0.3, 0.7))

        # Fallback to screener.in
        if not data:
            data = fetch_screener_fundamentals(nse_symbol)
            if data:
                time.sleep(random.uniform(0.5, 1.0))

        if data:
            data['ticker']     = ticker
            data['nse_symbol'] = nse_symbol
            data['fetched_at'] = datetime.now().strftime('%Y-%m-%d')
            records.append(data)
            success += 1
        else:
            failed += 1

        if (success + failed) % 10 == 0:
            print(f"  Progress: {success} success, {failed} failed / {len(tickers)} total")

    print(f"  Done: {success} fetched, {failed} failed")

    if records:
        df = pd.DataFrame(records)
        df['fetched_at'] = pd.to_datetime(df['fetched_at'])
        df.to_csv(CACHE_CSV, index=False)
        print(f"  Saved to {CACHE_CSV}")
        return df

    return pd.DataFrame()


# ── COMPUTE REAL QUALITY SCORE ────────────────────────────────────────
def compute_real_quality_score(fundamentals: pd.DataFrame, ticker: str) -> dict:
    """
    Compute true quality score using real fundamental data.
    Returns dict with score (0-100) and component breakdown.

    Scoring:
      ROE           — 0-30 pts (most important quality metric)
      Debt/Equity   — 0-25 pts (financial stability)
      Operating Margin — 0-25 pts (business quality)
      Current Ratio — 0-20 pts (short-term health)
    """
    row = fundamentals[fundamentals['ticker'] == ticker]
    if row.empty:
        return {'quality_real': None, 'quality_source': 'unavailable'}

    row = row.iloc[0]
    total = 0.0
    signals = {}

    # ROE (0-30 pts)
    roe = row.get('roe')
    if roe is not None and not pd.isna(roe):
        roe_pct = float(roe) * 100 if float(roe) < 2 else float(roe)  # Handle decimal vs %
        if roe_pct > 25:    pts = 30
        elif roe_pct > 18:  pts = 22
        elif roe_pct > 12:  pts = 15
        elif roe_pct > 6:   pts = 8
        else:               pts = 0
        total += pts
        signals['roe_pct'] = round(roe_pct, 1)
        signals['roe_pts'] = pts

    # Debt/Equity (0-25 pts, lower is better)
    de = row.get('debt_to_equity')
    if de is not None and not pd.isna(de):
        de = float(de)
        if de < 0.1:    pts = 25   # Nearly debt-free
        elif de < 0.3:  pts = 20
        elif de < 0.6:  pts = 14
        elif de < 1.0:  pts = 8
        elif de < 2.0:  pts = 3
        else:           pts = 0    # Highly leveraged
        total += pts
        signals['debt_equity'] = round(de, 2)
        signals['de_pts'] = pts

    # Operating Margin (0-25 pts)
    opm = row.get('operating_margin')
    if opm is not None and not pd.isna(opm):
        opm_pct = float(opm) * 100 if float(opm) < 2 else float(opm)
        if opm_pct > 25:    pts = 25
        elif opm_pct > 18:  pts = 20
        elif opm_pct > 12:  pts = 14
        elif opm_pct > 6:   pts = 7
        elif opm_pct > 0:   pts = 2
        else:               pts = 0   # Loss-making
        total += pts
        signals['opm_pct'] = round(opm_pct, 1)
        signals['opm_pts'] = pts

    # Current Ratio (0-20 pts)
    cr = row.get('current_ratio')
    if cr is not None and not pd.isna(cr):
        cr = float(cr)
        if cr > 2.0:    pts = 20
        elif cr > 1.5:  pts = 15
        elif cr > 1.0:  pts = 8    # Minimally solvent
        elif cr > 0.8:  pts = 3
        else:           pts = 0    # Liquidity risk
        total += pts
        signals['current_ratio'] = round(cr, 2)
        signals['cr_pts'] = pts

    return {
        'quality_real':   round(total, 1),
        'quality_source': 'fundamental',
        **signals
    }


# ── COMPUTE REAL EARNINGS SCORE ───────────────────────────────────────
def compute_real_earnings_score(fundamentals: pd.DataFrame, ticker: str) -> dict:
    """
    Compute true earnings quality score using real fundamental data.

    Scoring:
      Revenue Growth YoY  — 0-35 pts
      Net Margin          — 0-30 pts
      Free Cash Flow      — 0-20 pts (positive FCF = real business)
      ROA                 — 0-15 pts
    """
    row = fundamentals[fundamentals['ticker'] == ticker]
    if row.empty:
        return {'earnings_real': None, 'earnings_source': 'unavailable'}

    row = row.iloc[0]
    total = 0.0
    signals = {}

    # Revenue Growth (0-35 pts)
    rev_growth = row.get('revenue_growth_yoy')
    if rev_growth is None or pd.isna(rev_growth):
        # Estimate from performance data if available
        perf_1y = row.get('perf_1y')
        if perf_1y is not None and not pd.isna(perf_1y):
            rev_growth = float(perf_1y) / 100  # Very rough proxy
    if rev_growth is not None and not pd.isna(rev_growth):
        rg_pct = float(rev_growth) * 100 if abs(float(rev_growth)) < 5 else float(rev_growth)
        if rg_pct > 20:    pts = 35
        elif rg_pct > 12:  pts = 26
        elif rg_pct > 6:   pts = 18
        elif rg_pct > 0:   pts = 10
        elif rg_pct > -5:  pts = 4
        else:              pts = 0
        total += pts
        signals['revenue_growth_pct'] = round(rg_pct, 1)
        signals['rev_pts'] = pts

    # Net Margin (0-30 pts)
    nm = row.get('net_margin')
    if nm is not None and not pd.isna(nm):
        nm_pct = float(nm) * 100 if abs(float(nm)) < 2 else float(nm)
        if nm_pct > 20:    pts = 30
        elif nm_pct > 12:  pts = 22
        elif nm_pct > 6:   pts = 15
        elif nm_pct > 0:   pts = 7
        else:              pts = 0   # Loss-making
        total += pts
        signals['net_margin_pct'] = round(nm_pct, 1)
        signals['nm_pts'] = pts

    # Free Cash Flow (0-20 pts) — positive FCF is a key quality signal
    fcf       = row.get('free_cash_flow')
    net_income = row.get('net_income')
    if fcf is not None and net_income is not None and not pd.isna(fcf) and not pd.isna(net_income):
        fcf = float(fcf)
        ni  = float(net_income) if float(net_income) != 0 else 1
        fcf_quality = fcf / abs(ni)  # FCF as % of net income
        if fcf_quality > 0.8:   pts = 20   # FCF > 80% of earnings — high quality
        elif fcf_quality > 0.5: pts = 14
        elif fcf_quality > 0.2: pts = 8
        elif fcf_quality > 0:   pts = 4
        else:                   pts = 0    # Negative FCF
        total += pts
        signals['fcf_quality_ratio'] = round(fcf_quality, 2)
        signals['fcf_pts'] = pts

    # ROA (0-15 pts)
    roa = row.get('roa')
    if roa is not None and not pd.isna(roa):
        roa_pct = float(roa) * 100 if abs(float(roa)) < 2 else float(roa)
        if roa_pct > 15:   pts = 15
        elif roa_pct > 10: pts = 11
        elif roa_pct > 5:  pts = 7
        elif roa_pct > 0:  pts = 3
        else:              pts = 0
        total += pts
        signals['roa_pct'] = round(roa_pct, 1)
        signals['roa_pts'] = pts

    return {
        'earnings_real':   round(total, 1),
        'earnings_source': 'fundamental',
        **signals
    }


# ── BLEND REAL + PRICE SCORES ─────────────────────────────────────────
def blend_scores(price_score: float, real_score: Optional[float],
                 real_weight: float = 0.7) -> float:
    """
    Blend real fundamental score with price-based proxy.
    When real data is available, it gets 70% weight.
    When unavailable, falls back entirely to price proxy.
    """
    if real_score is None:
        return price_score
    return real_score * real_weight + price_score * (1 - real_weight)


# ── MAIN ──────────────────────────────────────────────────────────────
def run():
    print("\n" + "="*55)
    print("  FUNDAMENTAL DATA FETCHER")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*55 + "\n")

    from stock_screener import NIFTY500_UNIVERSE
    print(f"  Universe: {len(NIFTY500_UNIVERSE)} stocks\n")

    df = fetch_fundamentals_batch(NIFTY500_UNIVERSE)

    if df.empty:
        print("  No data fetched")
        return

    print("\n" + "="*55)
    print("  FUNDAMENTAL DATA SUMMARY")
    print("="*55)
    print(f"  Stocks with data: {len(df)}")

    # Show sample quality metrics
    for col, label in [('roe','ROE'), ('debt_to_equity','D/E'), ('operating_margin','OPM')]:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                median = valid.median()
                if col != 'debt_to_equity' and median < 2:
                    median *= 100  # Convert to %
                print(f"  Median {label}: {median:.1f}")

    print("\n  Top 5 by ROE:")
    if 'roe' in df.columns:
        top_roe = df[['ticker','roe']].dropna().sort_values('roe', ascending=False).head(5)
        for _, row in top_roe.iterrows():
            roe = float(row['roe'])
            if roe < 2: roe *= 100
            print(f"    {row['ticker'].replace('.NS',''):<15} ROE: {roe:.1f}%")

    print("\n  Lowest Debt/Equity (cleanest balance sheets):")
    if 'debt_to_equity' in df.columns:
        low_de = df[['ticker','debt_to_equity']].dropna().sort_values('debt_to_equity').head(5)
        for _, row in low_de.iterrows():
            print(f"    {row['ticker'].replace('.NS',''):<15} D/E: {float(row['debt_to_equity']):.2f}")

    print(f"\n  Cache saved: {CACHE_CSV}")
    print("="*55 + "\n")


if __name__ == "__main__":
    run()
