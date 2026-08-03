"""
fetch_data.py
Pulls the full current S&P 500 constituent list with GICS sector
labels from Wikipedia, then fetches daily close price and volume data
for the entire universe, for use in cross-sectional signal research.
"""

import io
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

START_DATE = "2019-01-01"
END_DATE = "2026-07-28"


def get_sp500_universe():
    """
    Scrapes the current S&P 500 constituent list from Wikipedia,
    including GICS sector classification.
    Returns a DataFrame with columns: Symbol, Sector.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    sp500_table = tables[0]

    universe = sp500_table[["Symbol", "GICS Sector"]].copy()
    universe.columns = ["Symbol", "Sector"]
    universe["Symbol"] = universe["Symbol"].str.replace(".", "-", regex=False)

    return universe


def fetch_universe_prices(tickers, start=START_DATE, end=END_DATE, batch_size=40, pause=2):
    """
    Downloads adjusted close prices and volume for a list of tickers,
    in batches to avoid overloading yfinance in a single call.
    Returns (close_df, volume_df).
    """
    all_close, all_volume = [], []
    n_batches = -(-len(tickers) // batch_size)  # ceiling division

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Fetching batch {batch_num} / {n_batches} ({len(batch)} tickers)...")

        try:
            data = yf.download(batch, start=start, end=end, auto_adjust=True)
            all_close.append(data["Close"])
            all_volume.append(data["Volume"])
        except Exception as e:
            print(f"Batch starting at index {i} failed: {e}")

        time.sleep(pause)

    close = pd.concat(all_close, axis=1)
    volume = pd.concat(all_volume, axis=1)

    # Drop any duplicate columns from batch overlaps
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    return close, volume


def compute_returns(price_df):
    return price_df.pct_change()


def save_data(universe, close, volume, returns):
    DATA_DIR.mkdir(exist_ok=True)
    universe.to_csv(DATA_DIR / "sp500_universe.csv", index=False)
    close.to_csv(DATA_DIR / "close.csv")
    volume.to_csv(DATA_DIR / "volume.csv")
    returns.to_csv(DATA_DIR / "returns.csv")
    print(f"Saved universe, close, volume, and returns to {DATA_DIR}/")


def main():
    universe = get_sp500_universe()
    print(f"Found {len(universe)} tickers across {universe['Sector'].nunique()} sectors.")

    close, volume = fetch_universe_prices(universe["Symbol"].tolist())
    returns = compute_returns(close)

    save_data(universe, close, volume, returns)
    print(returns.tail())


if __name__ == "__main__":
    main()