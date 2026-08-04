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

def momentum_reversion_sector_conditional_naive(close, universe, momentum_sectors, reversal_sectors,
                                                    momentum_lookback, reversal_lookback):
    """
    Composite signal that applies momentum in sectors where momentum
    historically outperforms, and mean reversion in sectors where
    reversal historically outperforms. Sectors not in either list are
    left NaN (excluded from the cross-sectional ranking).
    """
    momentum_signal = momentum_naive(close, lookback=momentum_lookback)
    reversal_signal = mean_reversion_naive(close, lookback=reversal_lookback)

    sector_map = dict(zip(universe["Symbol"], universe["Sector"]))

    composite = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    for ticker in close.columns:
        sector = sector_map.get(ticker)
        if sector in momentum_sectors:
            composite[ticker] = momentum_signal[ticker]
        elif sector in reversal_sectors:
            composite[ticker] = reversal_signal[ticker]

    return composite


SIGNAL_LIBRARY = {
    "Momentum": momentum_naive,
    "Short_Term_Reversal": mean_reversion_naive,
    "Volume_Spike": volume_spike,
    "Low_Volatility": low_volatility,
    "Momentum_Reversal_Hybrid": momentum_reversion_sector_conditional_naive
}


def compute_all_signals(close, volume, universe=None, momentum_sectors=None,
                          reversal_sectors=None, momentum_lookback=205,
                          reversal_lookback=48):
    """
    Computes every signal in the library. The sector-conditional
    composite is only included if `universe`, `momentum_sectors`, and
    `reversal_sectors` are all provided — otherwise it's skipped, since
    it needs sector metadata the other signals don't.
    """
    signals = {}
    signals["Momentum"] = momentum_naive(close)
    signals["Short_Term_Reversal"] = mean_reversion_naive(close)
    signals["Volume_Spike"] = volume_spike(volume)
    signals["Low_Volatility"] = low_volatility(close)

    if universe is not None and momentum_sectors is not None and reversal_sectors is not None:
        signals["Momentum_Reversal_Hybrid"] = momentum_reversion_sector_conditional_naive(
            close, universe,
            momentum_sectors=momentum_sectors,
            reversal_sectors=reversal_sectors,
            momentum_lookback=momentum_lookback,
            reversal_lookback=reversal_lookback,
        )

    return signals