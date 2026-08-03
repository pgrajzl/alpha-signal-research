"""
indicators.py
Technical indicator calculations for the stock explorer dashboard.
Mirrors the indicator library from the earlier stock-indicator-dashboard
project: SMA, EMA, Bollinger Bands (overlay on price), RSI, MACD,
Volume, OBV (sub-panel below price).
"""

import pandas as pd
import numpy as np


def add_sma(series, window=20):
    return series.rolling(window=window).mean()


def add_ema(series, window=20):
    return series.ewm(span=window, adjust=False).mean()


def add_bollinger_bands(series, window=20, num_std=2):
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def add_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_obv(close_series, volume_series):
    direction = np.sign(close_series.diff()).fillna(0)
    return (direction * volume_series).cumsum()


# Categorization: overlay indicators plot directly on the price panel;
# subpanel indicators get their own row below.
OVERLAY_INDICATORS = ["None", "SMA 20", "SMA 50", "EMA 20", "Bollinger Bands"]
SUBPANEL_INDICATORS = ["Volume", "OBV", "RSI", "MACD"]