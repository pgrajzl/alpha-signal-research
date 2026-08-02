"""
signals.py
Cross-sectional signal library. Each function takes price/volume
DataFrames (dates x tickers) and returns a DataFrame of the same shape
containing a signal score per stock per date — higher score = more
attractive (expected to outperform).
"""

import pandas as pd
import numpy as np


def momentum_12_1(close, lookback=252, skip=21):
    """
    Classic 12-1 month momentum: return over the past ~12 months,
    excluding the most recent month (to avoid short-term reversal
    contamination).
    """
    total_return = close.pct_change(lookback)
    recent_return = close.pct_change(skip)
    # 12-1 momentum = growth over the lookback period, excluding the
    # most recent `skip` days
    signal = (1 + total_return) / (1 + recent_return) - 1
    return signal


def short_term_reversal(close, lookback=5):
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
    "Momentum_12_1": momentum_12_1,
    "Short_Term_Reversal": short_term_reversal,
    "Volume_Spike": volume_spike,
    "Low_Volatility": low_volatility,
}


def compute_all_signals(close, volume):
    """Computes every signal in the library. Returns a dict of {name: DataFrame}."""
    signals = {}
    signals["Momentum_12_1"] = momentum_12_1(close)
    signals["Short_Term_Reversal"] = short_term_reversal(close)
    signals["Volume_Spike"] = volume_spike(volume)
    signals["Low_Volatility"] = low_volatility(close)
    return signals