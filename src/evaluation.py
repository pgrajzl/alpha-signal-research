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

import warnings
from scipy.stats import ConstantInputWarning

def compute_ic_series(signal_df, forward_returns_df):
    """
    For each date, computes the Spearman rank correlation between the
    signal's cross-sectional scores and the forward returns across all
    tickers. Returns a Series of IC values indexed by date.
    """

    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    
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

def compute_ic_by_sector(signal_df, close, universe, horizon, min_sector_size=5):
    """
    Computes mean IC separately within each GICS sector, to check
    whether a signal's effect is broad-based across the market or
    concentrated in a handful of sectors. `universe` is the DataFrame
    with Symbol/Sector columns from get_sp500_universe().
    """
    fwd_returns = compute_forward_returns(close, horizon=horizon)
    sector_map = dict(zip(universe["Symbol"], universe["Sector"]))

    sector_results = {}
    for sector in universe["Sector"].unique():
        sector_tickers = [t for t, s in sector_map.items()
                           if s == sector and t in signal_df.columns]
        if len(sector_tickers) < min_sector_size:
            continue

        ic_series = compute_ic_series(signal_df[sector_tickers], fwd_returns[sector_tickers])
        sector_results[sector] = {
            "mean_ic": ic_series.mean(),
            "n_stocks": len(sector_tickers),
        }

    return pd.DataFrame(sector_results).T.sort_values("mean_ic", ascending=False)

def compute_event_study_returns(signal_df, close, horizon=5):
    """
    For sparse, threshold-based signals, standard cross-sectional IC
    dilutes the effect since most stock-days score 0. This instead
    isolates every flagged event and computes forward returns only
    for those specific stock-date pairs, which is the more appropriate
    way to evaluate an infrequent-event signal.
    """
    forward_returns = compute_forward_returns(close, horizon=horizon)

    flagged = signal_df[signal_df != 0].stack()
    event_dates = flagged.index

    event_returns = []
    for date, ticker in event_dates:
        if date in forward_returns.index and ticker in forward_returns.columns:
            ret = forward_returns.loc[date, ticker]
            if pd.notna(ret):
                event_returns.append(ret)

    return pd.Series(event_returns)