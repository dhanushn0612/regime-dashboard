"""
FII/DII Data Converter & Appender
====================================
Converts NSE's downloaded CSV format into the format
the Regime Classifier needs, and appends to cumulative history.

USAGE:
    1. Download FII/DII CSV from nseindia.com/reports/fii-dii
    2. Save it anywhere (e.g. Downloads folder)
    3. Run: python data_pipeline/fii_dii_converter.py "C:\\Users\\Dhanush\\Downloads\\fii_dii.csv"

    Or run without argument to be prompted for the file path:
    python data_pipeline/fii_dii_converter.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(SCRIPT_DIR, 'fii_dii_data.csv')


def convert_nse_csv(input_path: str) -> pd.DataFrame:
    """
    Parse NSE's FII/DII CSV format:

    CATEGORY | DATE       | BUY VALUE | SELL VALUE | NET VALUE
    DII      | 13-Mar-26  | 21,407    | 12,055     | 9,351
    FII/FPI  | 13-Mar-26  | 11,320    | 21,440     | -10,119
    """
    print(f"\n  Reading: {input_path}")

    # Try reading with different encodings NSE uses
    df_raw = None
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            df_raw = pd.read_csv(input_path, encoding=enc)
            break
        except Exception:
            continue

    if df_raw is None:
        raise ValueError("Could not read the CSV file — try saving it as UTF-8 from Excel")

    print(f"  Raw columns: {list(df_raw.columns)}")
    print(f"  Raw rows: {len(df_raw)}")

    # Clean column names — strip whitespace and newlines
    df_raw.columns = [str(c).strip().replace('\n', ' ').replace('\r', '').upper() for c in df_raw.columns]

    # Find the right columns regardless of exact naming
    def find_col(df, keywords):
        for col in df.columns:
            if all(k.upper() in col.upper() for k in keywords):
                return col
        return None

    cat_col  = find_col(df_raw, ['CATEGORY'])
    date_col = find_col(df_raw, ['DATE'])
    net_col  = find_col(df_raw, ['NET'])

    if not all([cat_col, date_col, net_col]):
        print(f"  Available columns: {list(df_raw.columns)}")
        raise ValueError(f"Could not find required columns. Found: {list(df_raw.columns)}")

    print(f"  Using: category='{cat_col}', date='{date_col}', net='{net_col}'")

    # Clean the data
    df_raw[cat_col]  = df_raw[cat_col].astype(str).str.strip()
    df_raw[date_col] = df_raw[date_col].astype(str).str.strip()
    df_raw[net_col]  = df_raw[net_col].astype(str).str.replace(',', '').str.replace(' ', '').str.strip()

    # Drop empty rows
    df_raw = df_raw[df_raw[date_col].notna()]
    df_raw = df_raw[df_raw[date_col] != '']
    df_raw = df_raw[df_raw[date_col] != 'nan']

    # Parse dates — NSE uses formats like "13-Mar-26" or "13-Mar-2026" or "2026-03-13"
    def parse_date(d):
        for fmt in ['%d-%b-%y', '%d-%b-%Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']:
            try:
                return pd.to_datetime(d, format=fmt)
            except Exception:
                continue
        try:
            return pd.to_datetime(d, dayfirst=True)
        except Exception:
            return pd.NaT

    df_raw['parsed_date'] = df_raw[date_col].apply(parse_date)
    df_raw = df_raw[df_raw['parsed_date'].notna()]

    # Convert net values to float
    def parse_float(v):
        try:
            return float(str(v).replace(',', '').replace(' ', ''))
        except Exception:
            return None

    df_raw['net_float'] = df_raw[net_col].apply(parse_float)
    df_raw = df_raw[df_raw['net_float'].notna()]

    # Separate FII and DII rows
    fii_mask = df_raw[cat_col].str.upper().str.contains('FII|FPI')
    dii_mask = df_raw[cat_col].str.upper().str.contains('DII')

    fii_df = df_raw[fii_mask][['parsed_date', 'net_float']].rename(columns={'net_float': 'FII_Net'})
    dii_df = df_raw[dii_mask][['parsed_date', 'net_float']].rename(columns={'net_float': 'DII_Net'})

    # Merge on date
    merged = pd.merge(
        fii_df, dii_df,
        on='parsed_date', how='inner'
    ).rename(columns={'parsed_date': 'date'})

    merged['date'] = pd.to_datetime(merged['date'])
    merged = merged.sort_values('date').reset_index(drop=True)

    print(f"  Parsed {len(merged)} FII/DII records")
    if not merged.empty:
        print(f"  Date range: {merged['date'].min().strftime('%Y-%m-%d')} to {merged['date'].max().strftime('%Y-%m-%d')}")

    return merged


def append_to_history(new_df: pd.DataFrame) -> pd.DataFrame:
    """Append new records to cumulative history CSV."""

    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, parse_dates=['date'])
        print(f"\n  Existing history: {len(existing)} records")

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
        combined.sort_values('date', inplace=True)
        combined.reset_index(drop=True, inplace=True)

        new_records = len(combined) - len(existing)
        print(f"  Added {new_records} new records")
    else:
        combined = new_df
        print(f"\n  Created new history file with {len(combined)} records")

    # Save
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved to: {OUTPUT_CSV}")

    return combined


def print_summary(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"  FII/DII HISTORY SUMMARY")
    print(f"{'='*55}")
    print(f"  Total records : {len(df)} trading days")
    print(f"  Date range    : {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    fii20 = df['FII_Net'].tail(20).sum()
    dii20 = df['DII_Net'].tail(20).sum()

    print(f"\n  20-Day Rolling Flow:")
    print(f"  FII : {fii20:>+10,.0f} Cr  ({'BUYING ✓' if fii20 > 0 else 'SELLING ✗'})")
    print(f"  DII : {dii20:>+10,.0f} Cr  ({'BUYING ✓' if dii20 > 0 else 'SELLING ✗'})")

    print(f"\n  Last 5 Trading Days:")
    print(f"  {'Date':<13} {'FII Net':>12} {'DII Net':>12}")
    print(f"  {'-'*38}")
    for _, row in df.tail(5).iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d'):<13} {row['FII_Net']:>+12,.0f} {row['DII_Net']:>+12,.0f}")

    print(f"\n  Flow Interpretation:")
    if fii20 > 0 and dii20 > 0:
        print(f"  Both FII and DII buying — STRONG BULLISH signal")
    elif fii20 > 0 and dii20 < 0:
        print(f"  FII buying, DII selling — CAUTIOUSLY BULLISH")
    elif fii20 < 0 and dii20 > 0:
        print(f"  FII selling, DII supporting — MIXED signal")
    else:
        print(f"  Both FII and DII selling — BEARISH signal")

    print(f"{'='*55}\n")


def main():
    print(f"\n{'='*55}")
    print(f"  FII/DII CONVERTER — {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    # Get input file path
    if len(sys.argv) > 1:
        input_path = sys.argv[1].strip('"').strip("'")
    else:
        print("\n  Paste the full path to your downloaded NSE CSV file.")
        print("  Example: C:\\Users\\Dhanush\\Downloads\\fii_dii.csv")
        print()
        input_path = input("  File path: ").strip().strip('"').strip("'")

    if not os.path.exists(input_path):
        print(f"\n  ERROR: File not found: {input_path}")
        print(f"  Make sure the path is correct and the file exists.")
        return

    # Convert and append
    new_df   = convert_nse_csv(input_path)
    combined = append_to_history(new_df)
    print_summary(combined)

    print(f"  DONE. Your classifier will now use real FII/DII flow data.")
    print(f"  Run python data_pipeline\\run_classifier.py to update regime.\n")


if __name__ == "__main__":
    main()
