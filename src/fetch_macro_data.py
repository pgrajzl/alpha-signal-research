"""
fetch_macro_data.py
Pulls a broad set of macroeconomic series from FRED (Federal Reserve
Economic Data) for use in macro-driven signal research.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FRED_API_KEY = os.getenv("FRED_API_KEY")

# A broad set of widely-used macro series across categories:
# growth, inflation, labor, rates, credit, sentiment, housing
MACRO_SERIES = {
    # Growth
    "GDP": "Gross Domestic Product",
    "GDPC1": "Real GDP",
    "INDPRO": "Industrial Production Index",
    "RSAFS": "Retail Sales",

    # Inflation
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "CPILFESL": "Core CPI",
    "PCEPI": "PCE Price Index",
    "PCEPILFE": "Core PCE Price Index",

    # Labor
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "ICSA": "Initial Jobless Claims",
    "CIVPART": "Labor Force Participation Rate",

    # Interest rates / yield curve
    "DGS2": "2-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "DGS30": "30-Year Treasury Yield",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "FEDFUNDS": "Effective Federal Funds Rate",

    # Credit / financial conditions
    "BAMLH0A0HYM2": "High Yield Credit Spread",
    "BAA10Y": "Baa Corporate Bond Spread",
    "NFCI": "Chicago Fed National Financial Conditions Index",

    # Sentiment / volatility
    "UMCSENT": "U. Michigan Consumer Sentiment",
    "VIXCLS": "VIX",

    # Housing
    "HOUST": "Housing Starts",
    "CSUSHPISA": "Case-Shiller Home Price Index",

    # Money supply / liquidity
    "M2SL": "M2 Money Supply",

    # Dollar
    "DTWEXBGS": "Trade Weighted US Dollar Index",
}


def fetch_macro_series(series_dict=None, start="2000-01-01"):
    """
    Pulls every series in series_dict from FRED. Returns a DataFrame
    with one column per series, indexed by date. Series with different
    native frequencies (daily, weekly, monthly, quarterly) are left
    as-is here; resampling/alignment happens downstream when combining
    with equity data.
    """
    if FRED_API_KEY is None:
        raise ValueError(
            "FRED_API_KEY not found. Create a .env file in the project "
            "root with: FRED_API_KEY=your_key_here"
        )

    fred = Fred(api_key=FRED_API_KEY)

    series_dict = series_dict or MACRO_SERIES
    data = {}

    for code, description in series_dict.items():
        print(f"Fetching {code} ({description})...")
        try:
            series = fred.get_series(code, observation_start=start)
            data[code] = series
        except Exception as e:
            print(f"  Failed to fetch {code}: {e}")

    macro_df = pd.DataFrame(data)
    return macro_df


def save_macro_data(macro_df):
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "macro_data.csv"
    macro_df.to_csv(out_path)
    print(f"Saved macro data to {out_path}")


def main():
    macro_df = fetch_macro_series()
    save_macro_data(macro_df)
    print(macro_df.tail())


if __name__ == "__main__":
    main()