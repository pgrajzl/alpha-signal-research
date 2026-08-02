"""
fetch_data.py
Pulls a cross-sectional universe of liquid stocks (reusing the S&P 500
+ sector approach from prior projects) and daily OHLCV data for signal
research.
"""

import io
import time
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

START_DATE = "2019-01-01"
END_DATE = "2026-07-28"

# A liquid, diverse 40-stock universe across sectors (manageable size,
# avoids the multi-hour full S&P 500 pull from the pairs project)
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "CRM", "ADBE", "ORCL",
    "JPM", "BAC", "GS", "MS", "AIG", "V", "MA", "PYPL",
    "XOM", "CVX", "COP",
    "JNJ", "PFE", "UNH", "ABBV",
    "PG", "KO", "PEP", "WMT", "COST",
    "DUK", "CMS", "NEE",
    "CAT", "BA", "HON", "UPS",
    "DIS", "NFLX", "T"
]


def fetch_universe_prices(tickers=None, start=START_DATE, end=END_DATE, batch_size=20, pause=2):
    """Downloads adjusted close prices and volume for the universe."""
    tickers = tickers or UNIVERSE
    all_close, all_volume = [], []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Fetching batch {i // batch_size + 1} ({len(batch)} tickers)...")
        data = yf.download(batch, start=start, end=end, auto_adjust=True)
        all_close.append(data["Close"])
        all_volume.append(data["Volume"])
        time.sleep(pause)

    close = pd.concat(all_close, axis=1)
    volume = pd.concat(all_volume, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    return close, volume


def compute_returns(price_df):
    return price_df.pct_change()


def save_data(close, volume, returns):
    DATA_DIR.mkdir(exist_ok=True)
    close.to_csv(DATA_DIR / "close.csv")
    volume.to_csv(DATA_DIR / "volume.csv")
    returns.to_csv(DATA_DIR / "returns.csv")
    print(f"Saved close, volume, returns to {DATA_DIR}/")


def main():
    close, volume = fetch_universe_prices()
    returns = compute_returns(close)
    save_data(close, volume, returns)
    print(returns.tail())


if __name__ == "__main__":
    main()