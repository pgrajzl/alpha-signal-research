"""
signals.py
Cross-sectional signal library. Each function takes price/volume
DataFrames (dates x tickers) and returns a DataFrame of the same shape
containing a signal score per stock per date — higher score = more
attractive (expected to outperform).
"""

import pandas as pd
import numpy as np


def momentum_naive(close, lookback=63):
    """
    Simple price momentum: total return over the trailing `lookback`
    trading days, no skip period. Default lookback of 63 trading days
    (~3 months).
    """
    return close.pct_change(lookback)


def mean_reversion_naive(close, lookback=5):
    """
    Short-term reversal: stocks that fell recently tend to bounce.
    Signal is the NEGATIVE of the recent return (so a big recent drop
    produces a high, attractive score).
    """
    recent_return = close.pct_change(lookback)
    return -recent_return


def volume_spike(volume, window=20):
    """
    Volume spike: today's volume relative to its recent rolling average.
    High relative volume can signal informed trading / attention.
    """
    avg_volume = volume.rolling(window).mean()
    return volume / avg_volume - 1


def low_volatility(close, window=60):
    """
    Low volatility anomaly: less volatile stocks have historically
    delivered better risk-adjusted returns. Signal is NEGATIVE realized
    volatility (so lower vol = higher score).
    """
    returns = close.pct_change()
    realized_vol = returns.rolling(window).std()
    return -realized_vol


SIGNAL_LIBRARY = {
    "Momentum": momentum_naive,
    "Short_Term_Reversal": mean_reversion_naive,
    "Volume_Spike": volume_spike,
    "Low_Volatility": low_volatility,
}


def compute_all_signals(close, volume):
    signals = {}
    signals["Momentum"] = momentum_naive(close)
    signals["Short_Term_Reversal"] = mean_reversion_naive(close)
    signals["Volume_Spike"] = volume_spike(volume)
    signals["Low_Volatility"] = low_volatility(close)
    return signals