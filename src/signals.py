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

def extreme_drawdown_bounce_naive(close, drawdown_window=3, vol_lookback_days=20, threshold_std=2.0):
    """
    Flags stocks that experienced a statistically extreme drawdown
    (relative to their own recent volatility) over a short window, on
    the premise that extreme, panic-driven moves are more likely to
    mechanically overshoot and bounce back than routine moves.

    Unlike mean_reversion_naive (continuous, linear in the recent
    return), this only assigns a nonzero score to moves that cross a
    volatility-adjusted threshold — testing specifically whether
    extreme moves behave differently than garden-variety ones.

    drawdown_window: number of trading days over which cumulative
        drawdown is measured
    vol_lookback_days: number of trading days used to estimate each
        stock's own recent daily volatility
    threshold_std: how many standard deviations of daily vol the
        drawdown must exceed to be flagged as significant (default
        2 sigma)

    Returns a signal where flagged stocks get a positive score sized
    by how far beyond the threshold they fell, and all other stocks
    get 0.
    """
    # Cumulative return over the drawdown window
    cumulative_return = close.pct_change(drawdown_window)

    # Daily return volatility over the lookback period, scaled to
    # match the drawdown window (volatility scales with sqrt(time))
    daily_returns = close.pct_change()
    daily_vol = daily_returns.rolling(vol_lookback_days).std()
    window_vol = daily_vol * (drawdown_window ** 0.5)

    # z-score of the move relative to the stock's own typical volatility
    # over a window of this length
    z_score = cumulative_return / window_vol

    # Only flag drawdowns beyond -threshold_std; everything else gets 0.
    # Score scales with how far past the threshold the move went, so a
    # -4 sigma drop scores higher than a -2.1 sigma drop.
    is_extreme_drawdown = z_score <= -threshold_std
    signal = (-z_score - threshold_std).where(is_extreme_drawdown, 0)

    return signal

def rate_sensitivity_naive(returns, yield_10y, beta_window=60, yield_change_lookback=5):
    """
    Naive rate sensitivity signal: for each stock, estimates a rolling
    "rate beta" (sensitivity of the stock's daily returns to daily
    changes in the 10Y Treasury yield), then multiplies that beta by
    the recent cumulative change in the 10Y yield.

    returns: stock daily returns (dates x tickers)
    yield_10y: a Series of 10Y yield levels (e.g. macro_df["DGS10"])
    beta_window: rolling window (trading days) used to estimate rate beta
    yield_change_lookback: number of days over which the recent yield
        change is measured

    Note: rate beta is estimated via rolling covariance/variance
    (equivalent to a rolling OLS beta with the yield as the single
    regressor), which is far faster than looping a regression per
    ticker per date.
    """
    daily_yield_change = yield_10y.diff()
    aligned_daily_change = daily_yield_change.reindex(returns.index).ffill()

    recent_yield_move = yield_10y.diff(yield_change_lookback).reindex(returns.index).ffill()

    signal = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for ticker in returns.columns:
        stock_returns = returns[ticker]
        rolling_cov = stock_returns.rolling(beta_window).cov(aligned_daily_change)
        rolling_var = aligned_daily_change.rolling(beta_window).var()
        rate_beta = rolling_cov / rolling_var

        signal[ticker] = rate_beta * recent_yield_move

    return signal

def evaluate_rate_sensitivity(beta_window, yield_change_lookback, horizon,
                                 returns, yield_10y, close):
    """
    Evaluation wrapper for rate_sensitivity_naive, used for parallel
    grid search. Must live in a real module (not defined inline in a
    notebook) since multiprocessing on macOS uses spawn, which
    requires functions to be importable by module path.
    """
    from src.evaluation import compute_forward_returns, compute_ic_series, summarize_ic

    signal = rate_sensitivity_naive(
        returns, yield_10y,
        beta_window=beta_window,
        yield_change_lookback=yield_change_lookback,
    )
    fwd_returns = compute_forward_returns(close, horizon=horizon)
    ic_series = compute_ic_series(signal, fwd_returns)

    if len(ic_series) < 10:
        return {"mean_ic": float("nan"), "information_ratio": float("nan"), "n_obs": len(ic_series)}

    summary = summarize_ic(ic_series)
    return {
        "mean_ic": summary["Mean IC"],
        "information_ratio": summary["Information Ratio"],
        "n_obs": summary["N Observations"],
    }


SIGNAL_LIBRARY = {
    "Momentum": momentum_naive,
    "Short_Term_Reversal": mean_reversion_naive,
    "Volume_Spike": volume_spike,
    "Low_Volatility": low_volatility,
    "Momentum_Reversal_Hybrid": momentum_reversion_sector_conditional_naive,
    "Extreme_Drawdown_Bounce": extreme_drawdown_bounce_naive,
    "Rate_Sensitivity": rate_sensitivity_naive
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