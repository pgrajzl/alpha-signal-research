"""
evaluation.py
Computes the Information Coefficient (IC): the cross-sectional rank
correlation between a signal's scores and forward returns, for one or
more forward horizons. This is the standard way to measure whether a
signal actually predicts future performance.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def compute_forward_returns(close, horizon=5):
    """
    Forward return over `horizon` days, aligned so that the value on
    date T represents the return from T to T+horizon (i.e., what you'd
    earn holding a position entered on date T).
    """
    return close.pct_change(horizon).shift(-horizon)


def compute_ic_series(signal_df, forward_returns_df):
    """
    For each date, computes the Spearman rank correlation between the
    signal's cross-sectional scores and the forward returns across all
    tickers. Returns a Series of IC values indexed by date.
    """
    ic_values = {}
    for date in signal_df.index:
        sig_row = signal_df.loc[date].dropna()
        ret_row = forward_returns_df.loc[date].dropna()

        common = sig_row.index.intersection(ret_row.index)
        if len(common) < 5:  # need a reasonable cross-section to correlate
            continue

        ic, _ = spearmanr(sig_row[common], ret_row[common])
        ic_values[date] = ic

    return pd.Series(ic_values)


def summarize_ic(ic_series):
    """Standard IC summary stats: mean, std, IR (information ratio = mean/std)."""
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    ir = mean_ic / std_ic if std_ic != 0 else np.nan
    pct_positive = (ic_series > 0).mean()

    return {
        "Mean IC": mean_ic,
        "IC Std": std_ic,
        "Information Ratio": ir,
        "% Positive IC": pct_positive,
        "N Observations": len(ic_series),
    }


def compute_decay_curve(signal_df, close, horizons=(1, 5, 10, 20, 40)):
    """
    Computes mean IC at several forward horizons to show how a
    signal's predictive power decays over time.
    """
    decay = {}
    for h in horizons:
        fwd_returns = compute_forward_returns(close, horizon=h)
        ic_series = compute_ic_series(signal_df, fwd_returns)
        decay[h] = ic_series.mean()
    return pd.Series(decay)