# Regime-Adaptive Equity PMS — Methodology & Disclosures

## Strategy Overview

A rules-based Portfolio Management Service (PMS) that deploys capital
across Indian equities and fixed income instruments based on a 5-factor
market regime classifier. No discretionary decisions — every allocation
is determined by the regime score.

---

## 1. Regime Classifier (5 Dimensions)

The composite regime score (0–100) is computed daily from five dimensions:

| Dimension     | Weight | Source                          | Frequency |
|---------------|--------|---------------------------------|-----------|
| Trend         | 28%    | Nifty SMA50/200, momentum       | Daily     |
| Volatility    | 23%    | India VIX, realised vol         | Daily     |
| Breadth       | 22%    | % Nifty 200 stocks above 50 DMA | Daily     |
| Flow          | 17%    | FII/DII net flows (NSDL)        | Daily     |
| Yield Curve   | 10%    | India G-Sec 10Y-2Y spread       | Weekly    |

**Score → Regime → Equity Allocation:**

| Score   | Regime       | Equity | FI on Cash      |
|---------|-------------|--------|-----------------|
| ≥ 75    | Strong Bull  | 90%    | Liquid fund     |
| 55–74   | Mild Bull    | 80–88% | Liquid fund     |
| 40–54   | Neutral      | 60%    | Short duration  |
| 20–39   | Mild Bear    | 15%    | Liquid fund     |
| < 20    | Strong Bear  | 0%     | Overnight fund  |

**Fixed income rates:** RBI repo rate ± spread by instrument type.
Capital is never idle — non-equity allocation earns liquid/overnight
fund returns rather than sitting in cash.

---

## 2. Stock Selection

### Universe
- **Backtest:** 48 Nifty 100 large caps with price history to 2007
- **Live trading:** Nifty 500 (497 stocks, filtered by regime)
  - Bear/Neutral: Nifty 200 liquid names only (liquidity safety)
  - Bull: Full Nifty 500 (opportunity capture)

### Why 48 stocks for backtest?
Yahoo Finance provides reliable adjusted price data going back to 2005
for approximately 80–100 Indian large cap stocks. Using a fixed
universe of 48 Nifty 100 names — all of which were prominent large
caps throughout the backtest period — avoids introducing stocks that
gained prominence only recently. This is standard practice in Indian
systematic strategy backtesting.

**Survivorship bias note:** Mild survivorship bias exists because the
48 stocks were selected from today's Nifty 100 list. However, all 48
were major index constituents since at least 2005, making this bias
immaterial for the conclusions.

### Factor Model (per regime)

| Factor      | Bull weight | Neutral weight | Bear weight |
|-------------|-------------|----------------|-------------|
| Momentum    | 35%         | 30%            | 10%         |
| Quality     | 15%         | 20%            | 30%         |
| Low Vol     | 15%         | 25%            | 45%         |
| Earnings    | 25%         | 15%            | 10%         |
| Alt Data    | 10%         | 10%            | 5%          |

### Fundamental Data
- **Live:** Real ROE, D/E, operating margin, revenue growth from
  TradingView (current) and screener.in (historical quarterly)
- **Backtest pre-2018:** Price-based proxies for quality/earnings
- **Backtest 2018+:** Real annual fundamentals from screener.in
  with correct no-look-ahead dating
  (FY ending Mar Y available from Jun Y onwards)

---

## 3. Portfolio Construction

- **Weighting:** Ledoit-Wolf shrinkage covariance + Max Sharpe
  optimisation (scipy SLSQP)
- **Constraints:** Max 15% per stock, max 40% per sector
- **Turnover limit:** Max 30% monthly turnover
- **Transaction costs:** 20bps large cap, 40bps mid cap (round trip)

**Backtest vs live gap:** The backtest uses equal weighting within
the equity portion for simplicity. The live system uses optimised
weights. This is a known methodology gap that paper trading will
quantify.

---

## 4. Backtest Methodology

### Walk-Forward Design
- 18-year in-sample: Sep 2007 – Feb 2026 (222 months)
- Regime score re-computed each month using only past data
- No look-ahead bias in prices, fundamentals, or FII/DII data
- FII/DII: Real monthly data from Jan 2022; price-VIX proxy before

### Transaction Costs Applied
- 20bps round-trip for large caps (top 100)
- 40bps round-trip for mid caps
- Applied on entry and exit, not on holds
- Note: No bid-ask slippage model — add 10–15bps for conservative
  real-world estimate, reducing alpha by ~0.3–0.5% p.a.

### Fixed Income on Cash
- Applied in the "+FI" version of backtest only
- Cash portion earns instrument-appropriate rate from RBI repo history
- Overnight fund: repo – 10bps
- Liquid fund: at repo rate
- Short duration: repo + 25bps

---

## 5. Performance Summary (Sep 2007 – Feb 2026, 18 years)

### Without FI (equity + cash at 0%)
| Metric           | Strategy   | Nifty 50   |
|------------------|------------|------------|
| Total Return     | +855%      | +604%      |
| Annualised       | +13.0% p.a.| +11.1% p.a.|
| Sharpe           | 0.50       | 0.29       |
| Max Drawdown     | -19.1%     | -66.4%     |
| Calmar           | 0.68       | 0.17       |
| Win Rate         | 52%        | —          |
| Beta             | 0.31       | 1.00       |

### With FI (cash earns liquid/overnight fund rates)
| Metric           | Strategy+FI| Nifty 50   |
|------------------|------------|------------|
| Total Return     | ~+1,244%   | +604%      |
| Annualised       | ~+15.1%p.a.| +11.1% p.a.|
| Sharpe           | ~0.69      | 0.29       |
| Max Drawdown     | ~-16.1%    | -66.4%     |
| Calmar           | ~0.93      | 0.17       |

**FI adds approximately +2.3% p.a. additional return** from deploying
idle cash into liquid instruments rather than holding uninvested.

---

## 6. Key Assumptions & Limitations

1. **Equal weighting in backtest** — live system uses Ledoit-Wolf
   optimised weights. Actual live performance will differ.

2. **Liquid fund rates are approximations** — actual liquid fund
   returns track RBI repo closely but not exactly. Historical
   rates used are ± 10–15bps of actual.

3. **Pre-2018 fundamentals** — quality/earnings factors use
   price-based proxies before 2018 (when screener.in historical
   data begins). This slightly overstates quality signal reliability
   in the 2007–2017 period.

4. **No slippage model** — impact cost for mid/small caps is not
   modelled. Conservative estimate: deduct 10–15bps from transaction
   costs, reducing alpha by 0.3–0.5% p.a.

5. **FII/DII proxy pre-2022** — NSDL real data available from Jan
   2022 only. Earlier periods use a VIX-price proxy for the flow
   dimension (20% weight). The flow proxy explains ~60% of actual
   FII behavior — this is the weakest part of pre-2022 backtest.

6. **Fixed universe in backtest** — using today's Nifty 100
   composition for the full backtest period introduces mild
   survivorship bias. All 48 stocks were prominent large caps
   throughout the period, making this immaterial.

---

## 7. SEBI PMS Registration Requirements (India)

For Friends & Family capital (₹25L–1Cr tickets):
- SEBI PMS registration required for managing third-party capital
- Minimum ticket size: ₹50 lakh per investor
- Minimum net worth for manager: ₹50 crore
- Registration fee: ₹10 lakh + compliance infrastructure

**Alternative path for early stage:**
- Manage personal capital only (no registration required)
- Paper trade for 12–18 months to build live track record
- Register as PMS once track record and AUM justify it

---

## 8. Technology Stack

```
Data Pipeline (Python):
  run_classifier.py          — Daily regime score
  sector_rotation.py         — XGBoost sector signals (IC=0.325)
  stock_screener.py          — RF ranking with real fundamentals
  portfolio_construction.py  — Ledoit-Wolf + Max Sharpe
  risk_management.py         — 5 rules + Isolation Forest
  fixed_income.py            — FI deployment + yield curve
  regime_monitor.py          — Intra-month trigger monitoring
  paper_portfolio.py         — Paper trading tracker

Backtest:
  backtest_extended_20y.py   — 18-year walk-forward

Fundamental Data:
  build_fundamental_snapshots.py — Historical quarterly fundamentals
  fundamental_data.py            — Live fundamentals (TradingView)

Infrastructure:
  GitHub Actions — Daily automation (10 AM IST)
  Vercel         — Live dashboard hosting
  Dashboard      — regime-dashboard-nu.vercel.app
```

---

*Document version: March 2026*
*Author: Dhanush N, MSc Data Science, Madras School of Economics*
