"""
walk_forward.py
Splits a date range into sequential train/test windows for
walk-forward validation. Uses an expanding training window by default
(train on everything up to T, test on the next chunk), which mirrors
how you'd actually deploy a signal in production.
"""

import pandas as pd


def generate_walk_forward_windows(dates, train_years=2, test_months=6, expanding=True):
    """
    Returns a list of (train_start, train_end, test_start, test_end)
    tuples covering the full date range.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    start = dates.min()
    end = dates.max()

    windows = []
    train_start = start
    train_end = start + pd.DateOffset(years=train_years)

    while train_end < end:
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            test_end = end

        windows.append((train_start, train_end, test_start, test_end))

        if not expanding:
            train_start = train_start + pd.DateOffset(months=test_months)
        train_end = test_end

    return windows


def evaluate_signal_walk_forward(signal_df, close, horizon=5, train_years=2, test_months=6):
    """
    Runs IC evaluation on each walk-forward TEST window only (never
    train), then concatenates results into a single out-of-sample IC
    series covering the full period.
    """
    from src.evaluation import compute_forward_returns, compute_ic_series

    windows = generate_walk_forward_windows(signal_df.index, train_years, test_months)
    fwd_returns = compute_forward_returns(close, horizon=horizon)

    all_oos_ic = []
    for train_start, train_end, test_start, test_end in windows:
        test_signal = signal_df.loc[test_start:test_end]
        test_returns = fwd_returns.loc[test_start:test_end]

        ic_series = compute_ic_series(test_signal, test_returns)
        all_oos_ic.append(ic_series)

    return pd.concat(all_oos_ic).sort_index()