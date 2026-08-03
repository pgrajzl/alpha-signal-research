"""
combination.py
Combines multiple signals into a blended score. Weights are derived
from each signal's historical Information Coefficient during a
training period (IC-weighting) — this is the piece that actually
gives walk-forward validation something real to test, since the
weights themselves are "fit" on train data and then applied,
unseen, to test data.
"""

import pandas as pd
import numpy as np


def zscore_signal(signal_df):
    """Cross-sectionally z-scores a signal (per date, across tickers)."""
    mean = signal_df.mean(axis=1)
    std = signal_df.std(axis=1)
    return signal_df.sub(mean, axis=0).div(std, axis=0)


def compute_ic_weights(signals_dict, close, train_start, train_end, horizon=5):
    """
    Computes each signal's mean IC over the given training window,
    then converts those mean ICs into blend weights (higher historical
    IC -> higher weight). Negative or zero total IC across all signals
    falls back to equal weighting.
    """
    from src.evaluation import compute_forward_returns, compute_ic_series

    fwd_returns = compute_forward_returns(close, horizon=horizon)

    mean_ics = {}
    for name, sig_df in signals_dict.items():
        train_signal = sig_df.loc[train_start:train_end]
        train_returns = fwd_returns.loc[train_start:train_end]
        ic_series = compute_ic_series(train_signal, train_returns)
        mean_ics[name] = ic_series.mean()

    # Only give weight to signals with positive historical IC;
    # clip negatives to 0 so a historically bad signal doesn't get
    # included at all (rather than flipping sign, which risks overfitting
    # to noise in a short training window)
    positive_ics = {name: max(ic, 0) for name, ic in mean_ics.items()}
    total = sum(positive_ics.values())

    if total == 0:
        # No signal had positive historical IC -> fall back to equal weight
        n = len(signals_dict)
        return {name: 1 / n for name in signals_dict}, mean_ics

    weights = {name: ic / total for name, ic in positive_ics.items()}
    return weights, mean_ics


def combine_signals(signals_dict, weights=None):
    """
    Combines multiple signals (already-computed DataFrames) into one
    blended score via z-scoring each, then weighted averaging.
    If weights is None, falls back to equal weighting.
    """
    names = list(signals_dict.keys())
    weights = weights or {name: 1 / len(names) for name in names}

    zscored = {name: zscore_signal(df) for name, df in signals_dict.items()}
    combined = sum(zscored[name] * weights[name] for name in names)
    return combined


def walk_forward_combine(signals_dict, close, horizon=5, train_years=2, test_months=6):
    """
    Runs the full walk-forward combination: for each window, computes
    IC-based weights from the TRAIN period only, applies those weights
    to build a combined signal, and returns the combined signal
    restricted to each corresponding TEST period (concatenated across
    all windows). This is the version that genuinely uses train/test
    the way walk-forward validation is meant to.
    """
    from src.walk_forward import generate_walk_forward_windows

    # Use any one signal's index as the reference date range
    reference_dates = list(signals_dict.values())[0].index
    windows = generate_walk_forward_windows(reference_dates, train_years, test_months)

    combined_test_pieces = []
    weight_log = []

    for train_start, train_end, test_start, test_end in windows:
        weights, mean_ics = compute_ic_weights(
            signals_dict, close, train_start, train_end, horizon=horizon
        )
        weight_log.append({"test_start": test_start, "test_end": test_end, **weights})

        # Build the combined signal for the test period using train-derived weights
        test_signals = {name: df.loc[test_start:test_end] for name, df in signals_dict.items()}
        combined_test = combine_signals(test_signals, weights=weights)
        combined_test_pieces.append(combined_test)

    combined_oos = pd.concat(combined_test_pieces).sort_index()
    weight_log_df = pd.DataFrame(weight_log)

    return combined_oos, weight_log_df