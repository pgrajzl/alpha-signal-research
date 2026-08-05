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

import time

INTRADAY_START = "2026-07-01"
INTRADAY_END = "2026-08-05"  # yfinance end date is exclusive, so this captures through Aug 4 EOD


def fetch_intraday_data(tickers, start=INTRADAY_START, end=INTRADAY_END,
                          interval="5m", batch_size=10, pause=2):
    """
    Downloads intraday OHLCV bars for a list of tickers. Note: yfinance
    restricts 1-minute data to the trailing 7 days regardless of the
    requested range, so 5-minute bars are used by default to cover a
    full month-plus range while still being granular enough for VWAP.

    Returns a dict of {ticker: DataFrame}, since intraday data doesn't
    combine cleanly into one wide DataFrame the way daily close prices
    do (each ticker has its own timestamp index with gaps for non-
    trading hours).
    """
    import yfinance as yf

    intraday_data = {}
    n_batches = -(-len(tickers) // batch_size)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Fetching intraday batch {batch_num} / {n_batches} ({len(batch)} tickers)...")

        for ticker in batch:
            try:
                df = yf.download(ticker, start=start, end=end, interval=interval,
                                   auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    intraday_data[ticker] = df[["Open", "High", "Low", "Close", "Volume"]]
            except Exception as e:
                print(f"  Failed to fetch {ticker}: {e}")

        time.sleep(pause)

    return intraday_data


def compute_vwap(intraday_df):
    """
    Computes VWAP (volume-weighted average price) per trading day for
    a single ticker's intraday OHLCV DataFrame. VWAP resets each day
    (standard convention), using typical price (H+L+C)/3 weighted by
    volume.
    """
    df = intraday_df.copy()
    df["date"] = df.index.date
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["tp_volume"] = df["typical_price"] * df["Volume"]

    df["cum_tp_volume"] = df.groupby("date")["tp_volume"].cumsum()
    df["cum_volume"] = df.groupby("date")["Volume"].cumsum()
    df["vwap"] = df["cum_tp_volume"] / df["cum_volume"]

    return df.drop(columns=["date", "typical_price", "tp_volume", "cum_tp_volume", "cum_volume"])


def save_intraday_data(intraday_data, data_dir=None):
    data_dir = data_dir or DATA_DIR
    intraday_dir = data_dir / "intraday"
    intraday_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df in intraday_data.items():
        df.to_csv(intraday_dir / f"{ticker}_intraday.csv")

    print(f"Saved intraday data for {len(intraday_data)} tickers to {intraday_dir}/")