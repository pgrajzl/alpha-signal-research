"""
combination.py
Combines multiple signals into a single blended signal via
cross-sectional z-scoring and equal weighting.
"""

import pandas as pd


def zscore_signal(signal_df):
    """Cross-sectionally z-scores a signal (per date, across tickers)."""
    mean = signal_df.mean(axis=1)
    std = signal_df.std(axis=1)
    return signal_df.sub(mean, axis=0).div(std, axis=0)


def combine_signals(signals_dict, weights=None):
    """
    Combines multiple signals (already-computed DataFrames) into one
    blended score via z-scoring each, then weighted averaging.
    """
    names = list(signals_dict.keys())
    weights = weights or {name: 1 / len(names) for name in names}

    zscored = {name: zscore_signal(df) for name, df in signals_dict.items()}

    combined = sum(zscored[name] * weights[name] for name in names)
    return combined