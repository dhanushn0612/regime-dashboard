"""
Monthly FII/DII Converter for Backtest
========================================
Converts monthly FII/DII data (from NSE monthly reports)
into daily format the backtest can use.

Method: spreads monthly net flows evenly across trading days
in that month. This is a reasonable approximation for backtesting
since we don't have the exact daily breakdown.

Usage:
    python data_pipeline/fii_dii_monthly_converter.py
    (edit the DATA dict below with your monthly figures)

Output:
    data_pipeline/fii_dii_data.csv  — overwrites existing
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'fii_dii_data.csv')

# ── MONTHLY DATA (FII_Net and DII_Net in Crores) ─────────────────────
# Source: NSE India monthly FII/DII report
# Negative = net selling, Positive = net buying
MONTHLY_DATA = {
    'Jan-22': {'FII_Net': -41346.35, 'DII_Net': 21928.40},
    'Feb-22': {'FII_Net': -45720.07, 'DII_Net': 42084.07},
    'Mar-22': {'FII_Net': -43281.31, 'DII_Net': 39677.03},
    'Apr-22': {'FII_Net': -40652.71, 'DII_Net': 29869.52},
    'May-22': {'FII_Net': -54292.47, 'DII_Net': 50835.54},
    'Jun-22': {'FII_Net': -58112.37, 'DII_Net': 46599.23},
    'Jul-22': {'FII_Net':  -6567.71, 'DII_Net': 10546.02},
    'Aug-22': {'FII_Net':  22025.62, 'DII_Net': -7068.63},
    'Sep-22': {'FII_Net': -18308.30, 'DII_Net': 14119.75},
    'Oct-22': {'FII_Net':   -489.06, 'DII_Net':  9276.97},
    'Nov-22': {'FII_Net':  22546.34, 'DII_Net': -6301.32},
    'Dec-22': {'FII_Net': -14231.09, 'DII_Net': 24159.13},
    'Jan-23': {'FII_Net': -41464.73, 'DII_Net': 33411.85},
    'Feb-23': {'FII_Net': -11090.64, 'DII_Net': 19239.28},
    'Mar-23': {'FII_Net':   1997.70, 'DII_Net': 30548.77},
    'Apr-23': {'FII_Net':   5711.80, 'DII_Net':  2216.57},
    'May-23': {'FII_Net':  27856.48, 'DII_Net': -3306.35},
    'Jun-23': {'FII_Net':  27250.01, 'DII_Net':  4458.23},
    'Jul-23': {'FII_Net':  13922.01, 'DII_Net': -1184.33},
    'Aug-23': {'FII_Net': -20620.65, 'DII_Net': 25016.95},
    'Sep-23': {'FII_Net': -26692.16, 'DII_Net': 20312.65},
    'Oct-23': {'FII_Net': -29056.61, 'DII_Net': 25105.86},
    'Nov-23': {'FII_Net':   3901.82, 'DII_Net': 12720.36},
    'Dec-23': {'FII_Net':  31959.78, 'DII_Net': 12942.25},
    'Jan-24': {'FII_Net': -35977.87, 'DII_Net': 26743.63},
    'Feb-24': {'FII_Net': -15962.72, 'DII_Net': 25379.30},
    'Mar-24': {'FII_Net':   3314.47, 'DII_Net': 56311.60},
    'Apr-24': {'FII_Net': -35692.19, 'DII_Net': 44186.28},
    'May-24': {'FII_Net': -42214.28, 'DII_Net': 55733.04},
    'Jun-24': {'FII_Net':   2037.47, 'DII_Net': 28633.15},
    'Jul-24': {'FII_Net':   5407.83, 'DII_Net': 23486.02},
    'Aug-24': {'FII_Net': -20339.26, 'DII_Net': 50174.86},
    'Sep-24': {'FII_Net':  12611.79, 'DII_Net': 30857.30},
    'Oct-24': {'FII_Net':-114445.89, 'DII_Net':107254.68},
    'Nov-24': {'FII_Net': -45974.12, 'DII_Net': 44483.86},
    'Dec-24': {'FII_Net': -16982.48, 'DII_Net': 34194.73},
    'Jan-25': {'FII_Net': -87374.66, 'DII_Net': 86591.80},
    'Feb-25': {'FII_Net': -58988.08, 'DII_Net': 64853.19},
    'Mar-25': {'FII_Net':   2014.18, 'DII_Net': 37585.68},
    'Apr-25': {'FII_Net':   2735.02, 'DII_Net': 28228.45},
    'May-25': {'FII_Net':  11773.25, 'DII_Net': 67642.34},
    'Jun-25': {'FII_Net':   7488.98, 'DII_Net': 72673.91},
    'Jul-25': {'FII_Net': -47666.68, 'DII_Net': 60939.16},
    'Aug-25': {'FII_Net': -46902.92, 'DII_Net': 94828.55},
    'Sep-25': {'FII_Net': -35301.36, 'DII_Net': 65343.59},
    'Oct-25': {'FII_Net':  -2346.89, 'DII_Net': 52794.02},
    'Nov-25': {'FII_Net': -17500.31, 'DII_Net': 77083.78},
    'Dec-25': {'FII_Net': -34349.62, 'DII_Net': 79619.91},
    'Jan-26': {'FII_Net': -41435.22, 'DII_Net': 69220.74},
    'Feb-26': {'FII_Net':  -6640.78, 'DII_Net': 38423.11},
}


def expand_to_daily(monthly_data: dict) -> pd.DataFrame:
    """
    Spread monthly FII/DII totals evenly across trading days.
    This gives the backtest a daily series to compute rolling sums from.
    """
    records = []

    for month_str, values in monthly_data.items():
        try:
            month_date = pd.to_datetime(month_str, format='%b-%y')
        except Exception:
            continue

        # Get all trading days in this month (Mon-Fri)
        month_start = month_date.replace(day=1)
        month_end   = (month_date + pd.offsets.MonthEnd(0))

        # Generate business days
        bdays = pd.bdate_range(start=month_start, end=month_end)
        n_days = len(bdays)

        if n_days == 0:
            continue

        # Spread evenly
        daily_fii = values['FII_Net'] / n_days
        daily_dii = values['DII_Net'] / n_days

        for day in bdays:
            records.append({
                'date':    day.strftime('%Y-%m-%d'),
                'FII_Net': round(daily_fii, 2),
                'DII_Net': round(daily_dii, 2),
            })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def run():
    print("\n" + "="*55)
    print("  FII/DII MONTHLY CONVERTER")
    print("  " + datetime.today().strftime('%Y-%m-%d %H:%M'))
    print("="*55 + "\n")

    print(f"  Converting {len(MONTHLY_DATA)} months of FII/DII data...")
    df = expand_to_daily(MONTHLY_DATA)

    print(f"  Expanded to {len(df)} daily records")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Check for existing daily data and merge
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, parse_dates=['date'])
        # Keep existing daily data where available (more accurate)
        # Only fill in months not covered by daily data
        existing_dates = set(existing['date'].dt.strftime('%Y-%m-%d'))
        new_rows = df[~df['date'].dt.strftime('%Y-%m-%d').isin(existing_dates)]

        if len(new_rows) > 0:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.sort_values('date').reset_index(drop=True)
            combined.to_csv(OUTPUT_CSV, index=False)
            print(f"  Merged with existing: {len(combined)} total records")
        else:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"  Saved {len(df)} records")
    else:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  Saved {len(df)} records to {OUTPUT_CSV}")

    # Show summary
    print(f"\n  MONTHLY FII/DII SUMMARY:")
    print(f"  {'Month':<10} {'FII Net':>12} {'DII Net':>12} {'Signal'}")
    print(f"  {'-'*50}")
    for month, vals in list(MONTHLY_DATA.items())[-6:]:
        fii = vals['FII_Net']
        dii = vals['DII_Net']
        signal = "Both buying" if fii > 0 and dii > 0 else \
                 "FII selling, DII buying" if fii < 0 and dii > 0 else \
                 "Both selling" if fii < 0 and dii < 0 else "FII buying, DII selling"
        print(f"  {month:<10} {fii:>+12,.0f} {dii:>+12,.0f}  {signal}")

    print(f"\n  Done. Backtest will now use real FII/DII data for flow dimension.")
    print(f"  Run: python data_pipeline\\backtest.py\n")


if __name__ == "__main__":
    run()
