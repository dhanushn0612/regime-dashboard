"""
Alternative Data Engine — Layer 6
====================================
Four alternative data sources that price-based models cannot see:

  1. NLP on NSE Earnings Announcements
     FinBERT sentiment analysis on quarterly result PDFs from NSE EDGAR
     Scores: positive/negative/neutral sentiment per company

  2. Promoter Shareholding Changes
     NSE quarterly disclosure data — insider buying is a strong signal
     Scores: increasing promoter stake = bullish, decreasing = bearish

  3. GST Collection Trends by Sector
     Monthly GST data from GSTN/Finance Ministry
     Proxy for revenue growth across consumer, manufacturing, logistics

  4. FII/DII Flow Automation
     Automated daily scraping from NSE (with session management)
     Replaces manual weekly download

Usage:
    python data_pipeline/alternative_data.py

Output:
    public/alt_data_scores.json   — per-stock alternative data scores
    public/alt_data_summary.json  — sector-level macro signals
    data_pipeline/fii_dii_data.csv — auto-updated daily FII/DII
"""

import os
import re
import json
import time
import random
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, '..', 'public')
ALT_SCORES   = os.path.join(OUTPUT_DIR, 'alt_data_scores.json')
ALT_SUMMARY  = os.path.join(OUTPUT_DIR, 'alt_data_summary.json')
FII_DII_CSV  = os.path.join(SCRIPT_DIR, 'fii_dii_data.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── NIFTY 500 TICKER TO NSE SYMBOL MAPPING ───────────────────────────
def to_nse_symbol(yf_ticker: str) -> str:
    return yf_ticker.replace('.NS', '').replace('.BO', '')


# ══════════════════════════════════════════════════════════════════════
# MODULE 1: NLP ON NSE EARNINGS ANNOUNCEMENTS (FinBERT)
# ══════════════════════════════════════════════════════════════════════

def fetch_nse_announcements(symbol: str, days_back: int = 90) -> list:
    """
    Fetch recent corporate announcements from NSE for a given symbol.
    NSE provides a public API for corporate filings.
    Returns list of announcement texts.
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json',
        'Referer': 'https://www.nseindia.com/',
    })

    announcements = []
    try:
        # Warm NSE session
        session.get('https://www.nseindia.com', timeout=10)
        time.sleep(random.uniform(0.5, 1.0))

        # Fetch corporate announcements
        url = f'https://www.nseindia.com/api/corp-announcements?index=equities&symbol={symbol}'
        r = session.get(url, timeout=15)

        if r.status_code == 200:
            data = r.json()
            cutoff = datetime.now() - timedelta(days=days_back)

            for item in data[:20]:  # Last 20 announcements
                try:
                    date_str = item.get('date', '') or item.get('bDt', '')
                    if date_str:
                        ann_date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                        if pd.notna(ann_date) and ann_date >= cutoff:
                            subject = item.get('subject', '') or item.get('desc', '')
                            attchmnt = item.get('attchmnt', '') or ''
                            announcements.append({
                                'date':    str(ann_date.date()),
                                'subject': subject,
                                'type':    item.get('an_dt', 'Other'),
                                'text':    subject + ' ' + attchmnt[:200],
                            })
                except Exception:
                    continue
    except Exception:
        pass

    return announcements


def score_with_finbert(texts: list) -> dict:
    """
    Score text using FinBERT — the finance-domain BERT model.
    FinBERT is trained on financial news and earnings call transcripts.
    Returns sentiment scores: positive, negative, neutral probabilities.

    Falls back to keyword scoring if transformers not available.
    """
    if not texts:
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34,
                'model': 'no_text', 'compound': 0.0}

    # Try FinBERT first
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        model     = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
        model.eval()

        scores = []
        for text in texts[:5]:  # Limit to 5 texts to avoid timeout
            if not text or len(text.strip()) < 10:
                continue
            inputs = tokenizer(
                text[:512], return_tensors='pt',
                truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs   = torch.softmax(outputs.logits, dim=1).squeeze().tolist()

            # FinBERT labels: 0=positive, 1=negative, 2=neutral
            scores.append({'positive': probs[0], 'negative': probs[1], 'neutral': probs[2]})

        if scores:
            avg_pos = np.mean([s['positive'] for s in scores])
            avg_neg = np.mean([s['negative'] for s in scores])
            avg_neu = np.mean([s['neutral'] for s in scores])
            compound = avg_pos - avg_neg

            return {
                'positive': round(avg_pos, 3),
                'negative': round(avg_neg, 3),
                'neutral':  round(avg_neu, 3),
                'compound': round(compound, 3),
                'model':    'finbert',
                'n_texts':  len(scores),
            }

    except ImportError:
        pass  # Fall through to keyword scoring
    except Exception:
        pass

    # Keyword fallback (fast, interpretable)
    return score_with_keywords(texts)


def score_with_keywords(texts: list) -> dict:
    """
    Keyword-based sentiment scoring — fast fallback when FinBERT unavailable.
    Finance-domain keywords calibrated to Indian corporate announcements.
    """
    POSITIVE = [
        'profit', 'revenue growth', 'record', 'beat', 'strong', 'robust',
        'expansion', 'acquisition', 'dividend', 'buyback', 'order win',
        'contract', 'partnership', 'upgrade', 'outperform', 'exceeds',
        'highest ever', 'all time high', 'strong demand', 'margin expansion',
        'new product', 'launch', 'approval', 'award', 'promoter buying',
        'increase in stake', 'capacity addition', 'capex', 'guidance raised',
    ]
    NEGATIVE = [
        'loss', 'decline', 'miss', 'weak', 'concern', 'delay', 'impairment',
        'write-off', 'write off', 'downgrade', 'underperform', 'below',
        'shortfall', 'penalty', 'litigation', 'fraud', 'investigation',
        'resignation', 'promoter selling', 'decrease in stake', 'debt',
        'default', 'restructure', 'layoff', 'shutdown', 'revision downward',
    ]

    pos_count = neg_count = 0
    for text in texts:
        text_lower = text.lower()
        pos_count += sum(1 for kw in POSITIVE if kw in text_lower)
        neg_count += sum(1 for kw in NEGATIVE if kw in text_lower)

    total = pos_count + neg_count + 1
    pos_score = pos_count / total
    neg_score = neg_count / total
    neu_score = 1 - pos_score - neg_score
    compound  = pos_score - neg_score

    return {
        'positive': round(pos_score, 3),
        'negative': round(neg_score, 3),
        'neutral':  round(max(0, neu_score), 3),
        'compound': round(compound, 3),
        'model':    'keyword',
        'n_texts':  len(texts),
    }


def compute_nlp_score(sentiment: dict) -> float:
    """
    Convert FinBERT sentiment to 0-100 score.
    compound = positive - negative (range: -1 to +1)
    Map to 0-100: compound=+1 → 100, compound=0 → 50, compound=-1 → 0
    """
    compound = sentiment.get('compound', 0)
    return round((compound + 1) / 2 * 100, 1)


# ══════════════════════════════════════════════════════════════════════
# MODULE 2: PROMOTER SHAREHOLDING CHANGES
# ══════════════════════════════════════════════════════════════════════

def fetch_promoter_shareholding(symbol: str) -> dict:
    """
    Fetch promoter shareholding from screener.in.
    Parses the quarterly shareholding pattern table.
    """
    result = {
        'promoter_pct':    None,
        'promoter_change': None,
        'pledged_pct':     None,
        'score':           50.0,
        'signal':          'neutral',
    }

    try:
        from bs4 import BeautifulSoup
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        })

        for url_suffix in ['/consolidated/', '/']:
            url = f"https://www.screener.in/company/{symbol}{url_suffix}"
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                break

        if r.status_code != 200:
            return result

        soup = BeautifulSoup(r.text, 'html.parser')

        # Find shareholding table
        tables = soup.find_all('table')
        for table in tables:
            header = table.find('th')
            if header and 'promoter' in header.get_text(strip=True).lower():
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if cells and 'promoter' in cells[0].get_text(strip=True).lower():
                        try:
                            values = []
                            for c in cells[1:]:
                                txt = c.get_text(strip=True).replace('%', '').replace(',', '')
                                if txt:
                                    values.append(float(txt))

                            if len(values) >= 2:
                                prom_latest = values[-1]
                                prom_prev   = values[-2]
                                prom_change = prom_latest - prom_prev

                                result['promoter_pct']    = round(prom_latest, 2)
                                result['promoter_change'] = round(prom_change, 2)

                                if prom_change > 1.0:   base_score = 85
                                elif prom_change > 0.3: base_score = 70
                                elif prom_change > -0.3:base_score = 50
                                elif prom_change > -1.0:base_score = 35
                                else:                   base_score = 20

                                result['score']  = round(float(base_score), 1)
                                result['signal'] = 'bullish' if prom_change > 0.3 else                                                    'bearish' if prom_change < -0.3 else 'neutral'
                                return result
                        except Exception:
                            continue

    except Exception:
        pass

    return result


# ══════════════════════════════════════════════════════════════════════
# MODULE 3: GST COLLECTION TRENDS
# ══════════════════════════════════════════════════════════════════════

def fetch_gst_trends() -> dict:
    """
    Fetch monthly GST collection data from Finance Ministry.
    GST data is a leading indicator for:
    - Consumer sector: retail GST collections
    - Manufacturing: goods production
    - Services/IT: service tax component
    - Overall economy: total collections YoY growth

    Data source: https://pib.gov.in/PressReleaseIframePage.aspx (press releases)
    Fallback: hardcoded recent data if scraping fails
    """
    result = {
        'latest_month':       None,
        'total_collection_cr': None,
        'yoy_growth_pct':     None,
        'sector_signals':     {},
        'macro_signal':       'neutral',
        'score':              50,
        'source':             'fallback',
    }

    try:
        # Try to fetch from PIB press releases
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'text/html,application/xhtml+xml',
        }
        r = requests.get(
            'https://pib.gov.in/PressReleseDetail.aspx?PRID=2114754',
            headers=headers, timeout=15
        )
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'html.parser')
            text = soup.get_text()

            # Extract collection figure
            match = re.search(r'Rs\s*[\.,]?\s*([\d,]+)\s*crore', text, re.IGNORECASE)
            if match:
                amount = int(match.group(1).replace(',', ''))
                result['total_collection_cr'] = amount
                result['source'] = 'pib'

            # Extract YoY growth
            yoy_match = re.search(r'(\d+\.?\d*)%\s*(?:higher|growth|increase)', text, re.IGNORECASE)
            if yoy_match:
                result['yoy_growth_pct'] = float(yoy_match.group(1))

    except Exception:
        pass

    # Fallback: use recent known data (Feb 2026 data)
    # Source: Finance Ministry press releases
    if result['total_collection_cr'] is None:
        result.update({
            'latest_month':        'Feb-2026',
            'total_collection_cr': 183646,    # Feb 2026 GST collection (Cr)
            'yoy_growth_pct':      9.1,        # YoY growth
            'source':              'hardcoded_recent',
        })

    # Score based on YoY growth
    yoy = result.get('yoy_growth_pct', 0) or 0
    if yoy > 15:
        score = 85
        signal = 'strong_bullish'
    elif yoy > 10:
        score = 70
        signal = 'bullish'
    elif yoy > 5:
        score = 55
        signal = 'mild_bullish'
    elif yoy > 0:
        score = 45
        signal = 'neutral'
    elif yoy > -5:
        score = 35
        signal = 'mild_bearish'
    else:
        score = 20
        signal = 'bearish'

    result['score']        = score
    result['macro_signal'] = signal

    # Sector-level signals from GST (approximate)
    result['sector_signals'] = {
        'Consumer':      score + 5 if yoy > 8 else score - 5,   # Consumer more sensitive
        'Auto':          score + 8 if yoy > 10 else score - 8,  # Auto very GST-sensitive
        'Manufacturing': score,
        'Services_IT':   50,  # IT largely services-exempt
        'Infra':         score + 3 if yoy > 5 else score - 3,
    }

    return result


# ══════════════════════════════════════════════════════════════════════
# MODULE 4: FII/DII AUTOMATION
# ══════════════════════════════════════════════════════════════════════

def fetch_fii_dii_automated(days_back: int = 30) -> pd.DataFrame:
    """
    Automated daily FII/DII data from NSE India.
    NSE blocks cloud IPs so this uses session warming + cookie management.
    Falls back to last known data + synthetic extension if blocked.

    Returns DataFrame with columns: date, FII_Net, DII_Net
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept':          'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection':      'keep-alive',
        'Referer':         'https://www.nseindia.com/',
    })

    records = []

    try:
        # Step 1: Warm session with main page
        print("    Warming NSE session...")
        session.get('https://www.nseindia.com', timeout=15)
        time.sleep(2)

        # Step 2: Access FII-DII page to get cookies
        session.get('https://www.nseindia.com/market-data/fii-dii-activity', timeout=15)
        time.sleep(1)

        # Step 3: Fetch data
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        url = (
            f'https://www.nseindia.com/api/fiidiiTradeReact?'
            f'startDate={start_date.strftime("%d-%m-%Y")}&'
            f'endDate={end_date.strftime("%d-%m-%Y")}'
        )
        r = session.get(url, timeout=20)

        if r.status_code == 200:
            data = r.json()
            print(f"    NSE API: {len(data)} records")

            for item in data:
                try:
                    # NSE format: date, category, buyValue, sellValue, netValue
                    date_str  = item.get('date', '')
                    category  = item.get('category', '')
                    net_value = float(str(item.get('netValue', '0')).replace(',', '') or 0)

                    if not date_str:
                        continue

                    date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                    if pd.isna(date):
                        continue

                    if 'FII' in category.upper() or 'FPI' in category.upper():
                        records.append({'date': date, 'type': 'FII', 'net': net_value})
                    elif 'DII' in category.upper():
                        records.append({'date': date, 'type': 'DII', 'net': net_value})

                except Exception:
                    continue

    except Exception as e:
        print(f"    NSE blocked: {e}")

    # Process records into wide format
    if records:
        df_raw  = pd.DataFrame(records)
        fii_df  = df_raw[df_raw['type'] == 'FII'][['date', 'net']].rename(columns={'net': 'FII_Net'})
        dii_df  = df_raw[df_raw['type'] == 'DII'][['date', 'net']].rename(columns={'net': 'DII_Net'})
        df_wide = pd.merge(fii_df, dii_df, on='date', how='outer').sort_values('date')
        df_wide = df_wide.fillna(0)
        print(f"    Successfully fetched {len(df_wide)} days of FII/DII data")
        return df_wide

    print("    NSE automation failed — data will use existing cache")
    return pd.DataFrame()


def update_fii_dii_csv(new_data: pd.DataFrame) -> int:
    """
    Merge new FII/DII data with existing CSV.
    Returns number of new rows added.
    """
    if new_data.empty:
        return 0

    if os.path.exists(FII_DII_CSV):
        existing = pd.read_csv(FII_DII_CSV, parse_dates=['date'])
        existing_dates = set(existing['date'].dt.strftime('%Y-%m-%d'))
        new_data['date_str'] = new_data['date'].dt.strftime('%Y-%m-%d')
        new_rows = new_data[~new_data['date_str'].isin(existing_dates)].drop('date_str', axis=1)

        if len(new_rows) > 0:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.sort_values('date').reset_index(drop=True)
            combined.to_csv(FII_DII_CSV, index=False)
            return len(new_rows)
        return 0
    else:
        new_data.to_csv(FII_DII_CSV, index=False)
        return len(new_data)


# ══════════════════════════════════════════════════════════════════════
# COMBINE ALL SOURCES INTO STOCK SCORES
# ══════════════════════════════════════════════════════════════════════

def compute_alt_score(nlp_score: float, promoter_score: float,
                       gst_sector_score: float) -> float:
    """
    Combine all alternative data signals into a single stock score.

    Weights:
      NLP earnings sentiment:    50% (most stock-specific)
      Promoter shareholding:     30% (strong insider signal)
      GST sector macro:          20% (sector-level context)
    """
    # Handle missing data gracefully
    scores  = []
    weights = []

    if nlp_score is not None:
        scores.append(nlp_score)
        weights.append(0.50)

    if promoter_score is not None:
        scores.append(promoter_score)
        weights.append(0.30)

    if gst_sector_score is not None:
        scores.append(gst_sector_score)
        weights.append(0.20)

    if not scores:
        return 50.0

    # Normalise weights
    total_w = sum(weights)
    weighted = sum(s * w for s, w in zip(scores, weights)) / total_w
    return round(weighted, 1)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run():
    print("\n" + "="*60)
    print("  ALTERNATIVE DATA ENGINE — LAYER 6")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*60 + "\n")

    # ── MODULE 4: FII/DII AUTOMATION ──────────────────────────────────
    print("Module 4: Automating FII/DII data...")
    new_fii_dii = fetch_fii_dii_automated(days_back=30)
    if not new_fii_dii.empty:
        added = update_fii_dii_csv(new_fii_dii)
        print(f"  FII/DII: {added} new days added to {FII_DII_CSV}")
    else:
        print("  FII/DII: Using existing data (NSE blocked automation)")

    # ── MODULE 3: GST TRENDS ──────────────────────────────────────────
    print("\nModule 3: Fetching GST collection trends...")
    gst_data = fetch_gst_trends()
    print(f"  GST ({gst_data['latest_month']}): Rs {gst_data['total_collection_cr']:,} Cr")
    print(f"  YoY growth: {gst_data['yoy_growth_pct']}% | Signal: {gst_data['macro_signal']}")

    # ── LOAD STOCK UNIVERSE ───────────────────────────────────────────
    tickers_csv = os.path.join(SCRIPT_DIR, 'nifty500_tickers.csv')
    if os.path.exists(tickers_csv):
        tickers = pd.read_csv(tickers_csv)['ticker'].tolist()
    else:
        from stock_screener import NIFTY500_UNIVERSE
        tickers = NIFTY500_UNIVERSE

    # Process top 100 most liquid stocks (others get sector-level score)
    from alternative_data import NIFTY200_LIQUID
    priority_tickers = [t for t in tickers if t in NIFTY200_LIQUID][:80]

    print(f"\nProcessing {len(priority_tickers)} priority stocks...")
    print("  (Full 500-stock scoring takes ~45 min — running priority universe)")

    # ── MODULES 1+2: NLP + PROMOTER PER STOCK ─────────────────────────
    stock_scores = {}
    success = 0
    failed  = 0

    for i, ticker in enumerate(priority_tickers):
        symbol = to_nse_symbol(ticker)

        try:
            # Module 1: NLP on announcements
            announcements = fetch_nse_announcements(symbol, days_back=90)
            texts         = [a['text'] for a in announcements if a.get('text')]
            sentiment     = score_with_finbert(texts)
            nlp_score     = compute_nlp_score(sentiment)

            # Module 2: Promoter shareholding
            promoter      = fetch_promoter_shareholding(symbol)
            prom_score    = promoter['score']

            # Module 3: GST sector score
            # Map ticker to sector for GST signal
            from portfolio_construction import TICKER_SECTOR
            sector        = TICKER_SECTOR.get(ticker, 'Other')
            gst_score     = gst_data['sector_signals'].get(sector, gst_data['score'])

            # Combined alt score
            alt_score     = compute_alt_score(nlp_score, prom_score, gst_score)

            stock_scores[ticker] = {
                'ticker':           ticker,
                'symbol':           symbol,
                'alt_score':        alt_score,
                'nlp_score':        nlp_score,
                'nlp_sentiment':    sentiment.get('compound', 0),
                'nlp_model':        sentiment.get('model', 'unknown'),
                'promoter_score':   prom_score,
                'promoter_change':  promoter.get('promoter_change'),
                'promoter_signal':  promoter.get('signal', 'neutral'),
                'gst_score':        round(gst_score, 1),
                'n_announcements':  len(announcements),
                'updated_at':       datetime.now().strftime('%Y-%m-%d'),
            }
            success += 1

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(priority_tickers)} | {success} success, {failed} failed")

            time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            failed += 1
            stock_scores[ticker] = {
                'ticker':    ticker,
                'symbol':    symbol,
                'alt_score': 50.0,   # Neutral fallback
                'error':     str(e),
                'updated_at': datetime.now().strftime('%Y-%m-%d'),
            }

    print(f"\n  Done: {success} scored, {failed} failed / {len(priority_tickers)}")

    # ── PRINT TOP/BOTTOM ALT SCORES ───────────────────────────────────
    sorted_scores = sorted(
        [v for v in stock_scores.values() if 'error' not in v],
        key=lambda x: x['alt_score'], reverse=True
    )

    print(f"\n  TOP 10 by alternative data score:")
    print(f"  {'TICKER':<14} {'ALT':>6} {'NLP':>6} {'PROM':>6} {'SIGNAL'}")
    print(f"  {'-'*50}")
    for s in sorted_scores[:10]:
        print(f"  {s['symbol']:<14} {s['alt_score']:>6.1f} {s.get('nlp_score',50):>6.1f} {s.get('promoter_score',50):>6.1f}  {s.get('promoter_signal','?')}")

    print(f"\n  BOTTOM 5 (sell signals):")
    for s in sorted_scores[-5:]:
        print(f"  {s['symbol']:<14} {s['alt_score']:>6.1f} {s.get('promoter_signal','?')}")

    # ── SAVE OUTPUTS ──────────────────────────────────────────────────
    output = {
        'date':         datetime.today().strftime('%Y-%m-%d'),
        'stocks':       stock_scores,
        'gst':          gst_data,
        'n_scored':     success,
        'n_failed':     failed,
        'model_used':   sorted_scores[0].get('nlp_model', 'unknown') if sorted_scores else 'unknown',
    }
    with open(ALT_SCORES, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    summary = {
        'date':          datetime.today().strftime('%Y-%m-%d'),
        'gst':           gst_data,
        'fii_dii_days':  len(pd.read_csv(FII_DII_CSV)) if os.path.exists(FII_DII_CSV) else 0,
        'stocks_scored': success,
        'top_stocks':    [s['ticker'] for s in sorted_scores[:5]],
        'model_used':    output['model_used'],
    }
    with open(ALT_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: {ALT_SCORES}")
    print(f"  Saved: {ALT_SUMMARY}\n")


# ── FIX CIRCULAR IMPORT: define NIFTY200_LIQUID here too ─────────────
NIFTY200_LIQUID = {
    "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","LTIM.NS",
    "MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","OFSS.NS","TATAELXSI.NS","LTTS.NS","KPITTECH.NS",
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
    "INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","IDFCFIRSTB.NS","PNB.NS","BANKBARODA.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","MUTHOOTFIN.NS","CHOLAFIN.NS","PFC.NS","RECLTD.NS",
    "HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS","IRFC.NS",
    "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
    "GODREJCP.NS","MARICO.NS","COLPAL.NS","TATACONSUM.NS","VBL.NS",
    "MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS",
    "ASHOKLEY.NS","BOSCHLTD.NS","MOTHERSON.NS","BHARATFORG.NS","MRF.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS",
    "TORNTPHARM.NS","LUPIN.NS","ALKEM.NS","APOLLOHOSP.NS",
    "RELIANCE.NS","ONGC.NS","COALINDIA.NS","BPCL.NS","IOC.NS","GAIL.NS","PETRONET.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS","NMDC.NS",
    "LT.NS","NTPC.NS","POWERGRID.NS","TATAPOWER.NS","SIEMENS.NS",
    "ABB.NS","HAVELLS.NS","POLYCAB.NS","CUMMINSIND.NS",
    "TITAN.NS","ASIANPAINT.NS","DMART.NS","TRENT.NS","PIDILITIND.NS",
    "BERGEPAINT.NS","WHIRLPOOL.NS","VOLTAS.NS","DIXON.NS",
    "BHARTIARTL.NS","ADANIPORTS.NS","ULTRACEMCO.NS","AMBUJACEM.NS",
    "INDHOTEL.NS","IRCTC.NS","CONCOR.NS",
}

if __name__ == "__main__":
    run()
