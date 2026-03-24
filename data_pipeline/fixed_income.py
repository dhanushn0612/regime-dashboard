"""
Fixed Income & Yield Curve Engine
===================================
Components:
  1. Fixed Income deployment (cash earns liquid/overnight fund returns)
  2. Yield Curve signal (5th dimension in regime composite)
  3. FI-adjusted backtest results

Usage:
    python data_pipeline/fixed_income.py
    python data_pipeline/fixed_income.py --fi-only
    python data_pipeline/fixed_income.py --yc-only
"""

import os, re, sys, json, warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR  = os.path.join(SCRIPT_DIR, "..", "public")
YC_OUT      = os.path.join(PUBLIC_DIR, "yield_curve.json")
FI_OUT      = os.path.join(PUBLIC_DIR, "fi_backtest.json")
YC_HIST     = os.path.join(SCRIPT_DIR, "yc_history.csv")
BT_JSON     = os.path.join(PUBLIC_DIR, "backtest_results.json")
os.makedirs(PUBLIC_DIR, exist_ok=True)

# ── Historical RBI repo rates ─────────────────────────────────────────
RBI_REPO = {
    2005:6.00, 2006:6.50, 2007:7.50, 2008:6.50, 2009:4.75,
    2010:6.25, 2011:8.25, 2012:8.00, 2013:7.75, 2014:8.00,
    2015:6.75, 2016:6.25, 2017:6.00, 2018:6.50, 2019:5.15,
    2020:4.00, 2021:4.00, 2022:5.90, 2023:6.50, 2024:6.50,
    2025:6.25, 2026:6.00,
}

def fi_rate(year, code):
    """Annual FI rate based on year and regime code."""
    repo = RBI_REPO.get(year, 6.50) / 100
    if code == 0: return repo - 0.001   # overnight
    if code == 1: return repo           # liquid
    if code == 2: return repo + 0.0025  # short duration
    return repo - 0.0005                # residual cash in bull

def fi_instrument(code):
    return {0:"overnight", 1:"liquid", 2:"short_duration"}.get(code, "liquid")

# ── FI Backtest ───────────────────────────────────────────────────────
def run_fi_backtest():
    if not os.path.exists(BT_JSON):
        print("  backtest_results.json not found — run backtest first")
        return {}
    with open(BT_JSON) as f:
        bt = json.load(f)
    monthly = bt["monthly"]
    n = len(monthly)

    orig_rets, adj_rets, nifty_rets, fi_rets = [], [], [], []
    by_code = {i: [] for i in range(5)}

    for m in monthly:
        year    = int(str(m["date"])[:4])
        code    = m.get("regime_code", 2)
        eq      = float(m.get("equity_alloc", 0.60))
        cash    = 1.0 - eq
        er      = m.get("port_ret", 0.0) / 100
        nr      = m.get("nifty_ret", 0.0) / 100
        fi_m    = (fi_rate(year, code) / 12) * cash
        tr      = er + fi_m

        orig_rets.append(er)
        adj_rets.append(tr)
        nifty_rets.append(nr)
        fi_rets.append(fi_m * 100)
        by_code[code].append({"fi": fi_m * 100, "cash": cash})

    def metrics(rets, label):
        a = np.array(rets)
        cum = float(np.prod(1 + a) - 1)
        ann = float((1 + cum) ** (12/n) - 1)
        rf  = 0.065 / 12
        exc = a - rf
        sh  = float(exc.mean()/exc.std()*np.sqrt(12)) if exc.std()>0 else 0
        cs  = pd.Series(np.cumprod(1+a))
        hwm = cs.cummax()
        dd  = float(((cs-hwm)/hwm).min()) * 100
        cal = (ann * 100) / abs(dd) if dd != 0 else 0
        wr  = float((a > 0).mean() * 100)
        return dict(label=label, total=round(cum*100,2), ann=round(ann*100,2),
                    sharpe=round(sh,2), max_dd=round(dd,2),
                    calmar=round(cal,2), win_rate=round(wr,1))

    o = metrics(orig_rets, "Strategy (cash=0%)")
    f = metrics(adj_rets,  "Strategy + FI")
    nf= metrics(nifty_rets,"Nifty")

    labels = {0:"Strong Bear",1:"Mild Bear",2:"Neutral",3:"Mild Bull",4:"Strong Bull"}
    by_regime = {}
    for code, rows in by_code.items():
        if rows:
            avg_fi   = np.mean([r["fi"] for r in rows])
            avg_cash = np.mean([r["cash"] for r in rows])
            by_regime[labels[code]] = dict(
                months=len(rows),
                avg_cash_pct=round(avg_cash*100,1),
                avg_fi_annual=round(avg_fi*12,2),
                instrument=fi_instrument(code)
            )

    return dict(
        computed_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_months=n,
        original=o, fi_adjusted=f, nifty=nf,
        fi_contribution=dict(
            ann_return_added=round(f["ann"]-o["ann"],2),
            sharpe_improvement=round(f["sharpe"]-o["sharpe"],2),
            calmar_improvement=round(f["calmar"]-o["calmar"],2),
        ),
        by_regime=by_regime
    )

# ── Yield Curve Fetching ──────────────────────────────────────────────
def fetch_yields():
    yields = {}
    try:
        r = requests.get(
            "https://www.investing.com/rates-bonds/india-government-bonds",
            headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.investing.com/"},
            timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 2: continue
                name = cells[0].get_text(strip=True)
                for col in [1, 2]:
                    raw = cells[col].get_text(strip=True).replace("%","").replace(",",".").strip()
                    try:
                        rate = float(raw)
                        if not (5.0 <= rate <= 10.0): continue
                        if   "3M" in name:  yields["3m"]  = rate
                        elif "6M" in name:  yields["6m"]  = rate
                        elif "1Y" in name:  yields["1y"]  = rate
                        elif "2Y" in name:  yields["2y"]  = rate
                        elif "5Y" in name:  yields["5y"]  = rate
                        elif "10Y" in name: yields["10y"] = rate
                        break
                    except: continue
            if yields.get("10y") and 5.0 <= yields["10y"] <= 10.0:
                yields["source"] = "investing.com"
                return yields
    except: pass

    # Fallback: March 2026 actual values
    # 10Y G-Sec: 6.68% (post RBI 6% repo, flat curve)
    yields = {"3m":6.38,"6m":6.48,"1y":6.55,"2y":6.62,
              "5y":6.70,"10y":6.68,"source":"fallback_march_2026"}
    print("  Yield curve: using fallback (live fetch unavailable)")
    return yields

def current_repo(): return 6.00  # Feb 2026 cut to 6.00%

# ── Yield Curve Signal ────────────────────────────────────────────────
def yc_signal(yields, history=None):
    y10  = yields.get("10y", 6.68)
    y2   = yields.get("2y",  6.62)
    y3m  = yields.get("3m",  6.38)
    repo = current_repo()

    sp_10_2 = round(y10 - y2, 3)
    sp_10_3m= round(y10 - y3m, 3)
    pol_gap = round(y3m - repo, 3)

    s = 0
    # Steepness (30 pts)
    if   sp_10_2 >  1.00: s += 30
    elif sp_10_2 >  0.50: s += 24
    elif sp_10_2 >  0.15: s += 16
    elif sp_10_2 >  0.00: s += 8
    elif sp_10_2 > -0.25: s += 3

    # Policy (25 pts) - negative = cuts expected = bullish
    if   pol_gap < -0.50: s += 25
    elif pol_gap < -0.25: s += 20
    elif pol_gap <  0.00: s += 14
    elif pol_gap <  0.25: s += 8
    elif pol_gap <  0.50: s += 3

    # Level (20 pts)
    if   y10 < 6.00: s += 20
    elif y10 < 6.50: s += 16
    elif y10 < 7.00: s += 10
    elif y10 < 7.50: s += 4

    # Trend (25 pts)
    trend_s = 12
    if history is not None and len(history) >= 4:
        try:
            prev = float(history["spread_10_2"].dropna().iloc[-4])
            chg  = sp_10_2 - prev
            if   chg >  0.30: trend_s = 25
            elif chg >  0.10: trend_s = 18
            elif chg > -0.10: trend_s = 12
            elif chg > -0.30: trend_s = 5
            else:             trend_s = 0
        except: pass
    s += trend_s
    s = max(0, min(100, s))

    regime = ("Expansion" if s>=70 else "Recovery" if s>=55 else
              "Neutral" if s>=40 else "Slowdown" if s>=25 else "Contraction")

    if   sp_10_2 >  0.50: sectors = {k:"bullish" for k in ["Banks","Financials","Industrials"]}; sectors.update({k:"neutral" for k in ["IT","FMCG","Pharma"]})
    elif sp_10_2 <  0.00: sectors = {k:"bearish" for k in ["Banks","Financials","Industrials"]}; sectors.update({k:"bullish" for k in ["IT","FMCG","Pharma"]})
    else:                  sectors = {k:"neutral" for k in ["Banks","IT","FMCG","Pharma","Industrials","Financials"]}

    parts = []
    if   sp_10_2 >  0.50: parts.append(f"Steep ({sp_10_2:+.2f}%) — expansion")
    elif sp_10_2 >  0.15: parts.append(f"Positive slope ({sp_10_2:+.2f}%) — neutral")
    elif sp_10_2 >  0.00: parts.append(f"Nearly flat ({sp_10_2:+.2f}%) — late cycle")
    else:                  parts.append(f"Inverted ({sp_10_2:+.2f}%) — recession warning")
    if pol_gap < -0.25:    parts.append(f"Cut expectations ({-pol_gap:.2f}% below repo) — liquidity supportive")
    elif pol_gap > 0.25:   parts.append(f"Hike expectations ({pol_gap:.2f}% above repo) — tightening")
    if y10 > 7.20:         parts.append(f"High yields ({y10:.2f}%) — P/E headwind")
    elif y10 < 6.20:       parts.append(f"Low yields ({y10:.2f}%) — valuation supportive")

    return dict(
        date=date.today().strftime("%Y-%m-%d"),
        yields={k:v for k,v in yields.items() if k!="source"},
        repo_rate=repo, source=yields.get("source","unknown"),
        spreads={"10y_2y":sp_10_2,"10y_3m":sp_10_3m,"policy_gap":pol_gap},
        yc_score=s, yc_regime=regime,
        sector_signals=sectors, interpretation=" | ".join(parts)
    )

# ── 5-Dimension Composite ─────────────────────────────────────────────
def composite_5d(trend, vola, breadth, flow, yc):
    """Compute composite score including YC as 5th dimension."""
    W = dict(trend=0.28, vola=0.23, breadth=0.22, flow=0.17, yc=0.10)
    c = trend*W["trend"]+vola*W["vola"]+breadth*W["breadth"]+flow*W["flow"]+yc*W["yc"]
    c = round(c, 1)
    code = 4 if c>=75 else 3 if c>=55 else 2 if c>=40 else 1 if c>=20 else 0
    labels = {4:"Strong Bull",3:"Mild Bull",2:"Neutral",1:"Mild Bear",0:"Strong Bear"}
    return dict(composite=c, code=code, label=labels[code], weights=W)

# ── History ───────────────────────────────────────────────────────────
def update_history(yc):
    row = dict(date=yc["date"],
               y3m=yc["yields"].get("3m"), y2y=yc["yields"].get("2y"),
               y5y=yc["yields"].get("5y"), y10y=yc["yields"].get("10y"),
               repo=yc["repo_rate"], spread_10_2=yc["spreads"]["10y_2y"],
               spread_10_3m=yc["spreads"]["10y_3m"],
               policy_gap=yc["spreads"]["policy_gap"],
               yc_score=yc["yc_score"], yc_regime=yc["yc_regime"],
               source=yc["source"])
    if os.path.exists(YC_HIST):
        df = pd.read_csv(YC_HIST)
        if yc["date"] not in df["date"].values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(YC_HIST, index=False)
    else:
        pd.DataFrame([row]).to_csv(YC_HIST, index=False)

# ── Print ─────────────────────────────────────────────────────────────
def print_fi(fi):
    if not fi: return
    o,f,n,c = fi["original"],fi["fi_adjusted"],fi["nifty"],fi["fi_contribution"]
    print(f"\n  {'Metric':<28} {'Cash=0%':>9} {'With FI':>9} {'Nifty':>9}")
    print(f"  {'─'*52}")
    for label, vals in [("Total Return (%)",(o["total"],f["total"],n["total"])),
                        ("Annualised (%)",  (o["ann"],  f["ann"],  n["ann"])),
                        ("Sharpe",          (o["sharpe"],f["sharpe"],n["sharpe"])),
                        ("Max Drawdown (%)",(o["max_dd"],f["max_dd"],n["max_dd"])),
                        ("Calmar",          (o["calmar"],f["calmar"],n["calmar"]))]:
        print(f"  {label:<28} {vals[0]:>9.2f} {vals[1]:>9.2f} {vals[2]:>9.2f}")
    print(f"\n  FI adds: +{c['ann_return_added']:.2f}% p.a. | "
          f"Sharpe +{c['sharpe_improvement']:.2f} | "
          f"Calmar +{c['calmar_improvement']:.2f}")
    print(f"\n  By regime:")
    for regime, d in fi["by_regime"].items():
        print(f"  {regime:<14} {d['months']:>3}M | "
              f"cash={d['avg_cash_pct']:.0f}% → {d['instrument']} @ {d['avg_fi_annual']:.2f}% p.a.")

def print_yc(yc):
    print(f"\n  10Y G-Sec:    {yc['yields'].get('10y','N/A'):.2f}%  "
          f"| Repo: {yc['repo_rate']:.2f}%")
    print(f"  10Y-2Y spread:{yc['spreads']['10y_2y']:+.2f}%  "
          f"| Policy gap: {yc['spreads']['policy_gap']:+.2f}%")
    print(f"  YC Score: {yc['yc_score']}/100  |  Regime: {yc['yc_regime']}")
    print(f"  Source: {yc['source']}")
    print(f"\n  {yc['interpretation']}")
    print(f"\n  Sectors:")
    for sec, sig in yc["sector_signals"].items():
        dot = "🟢" if sig=="bullish" else "🔴" if sig=="bearish" else "⚪"
        print(f"    {dot} {sec}")

# ── Main ──────────────────────────────────────────────────────────────
def run():
    fi_only = "--fi-only" in sys.argv
    yc_only = "--yc-only" in sys.argv
    print(f"\n{'='*60}")
    print(f"  FIXED INCOME & YIELD CURVE ENGINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    if not yc_only:
        print("\nPART 1 — Fixed Income Backtest")
        fi = run_fi_backtest()
        if fi:
            print_fi(fi)
            with open(FI_OUT,"w") as f: json.dump(fi,f,indent=2,default=str)
            print(f"\n  Saved: {FI_OUT}")

    if not fi_only:
        print("\nPART 2 — Yield Curve Analysis")
        hist = pd.read_csv(YC_HIST) if os.path.exists(YC_HIST) else None
        yields = fetch_yields()
        yc = yc_signal(yields, hist)
        print_yc(yc)
        enh = composite_5d(55, 50, 45, 40, yc["yc_score"])
        old = 0.30*55+0.25*50+0.25*45+0.20*40
        print(f"\n  5D composite example: {old:.1f} → {enh['composite']:.1f} ({enh['label']})")
        update_history(yc)
        with open(YC_OUT,"w") as f: json.dump(yc,f,indent=2,default=str)
        print(f"\n  Saved: {YC_OUT}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    run()
