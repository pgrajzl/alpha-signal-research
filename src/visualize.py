"""
visualize.py
Plots for signal evaluation (IC time series, decay curves, cumulative
IC) plus an interactive dashboard for exploring any stock in the
universe over a selectable time range.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ipywidgets as widgets
from IPython.display import display
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


# ---------------------------------------------------------------------
# Signal evaluation plots
# ---------------------------------------------------------------------

def plot_ic_series(ic_series, signal_name):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(ic_series.index, ic_series.values, color="steelblue", width=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(ic_series.mean(), color="red", linestyle="--", linewidth=1,
               label=f"Mean IC = {ic_series.mean():.3f}")
    ax.set_title(f"{signal_name} — Information Coefficient Over Time")
    ax.set_ylabel("IC")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"ic_series_{signal_name}.png", dpi=150)
    plt.show()


def plot_decay_curves(decay_dict):
    """decay_dict: {signal_name: decay_series (indexed by horizon)}"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, decay_series in decay_dict.items():
        ax.plot(decay_series.index, decay_series.values, marker="o", label=name)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Signal Decay: Mean IC vs. Forward Horizon")
    ax.set_xlabel("Forward Horizon (days)")
    ax.set_ylabel("Mean IC")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / "decay_curves.png", dpi=150)
    plt.show()


def plot_cumulative_ic(ic_series, signal_name):
    cumulative = ic_series.cumsum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(cumulative.index, cumulative.values, color="darkgreen", linewidth=1.3)
    ax.set_title(f"{signal_name} — Cumulative IC")
    ax.set_ylabel("Cumulative IC")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"cumulative_ic_{signal_name}.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------
# Interactive stock explorer: ticker dropdown + date range dropdown
# ---------------------------------------------------------------------

RANGE_OPTIONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "YTD": "ytd",
    "1Y": 252,
    "3Y": 756,
    "All": None,
}


def _filter_by_range(series, range_label):
    if range_label == "All" or RANGE_OPTIONS[range_label] is None:
        return series
    if range_label == "YTD":
        start = series.index[series.index.year == series.index.max().year][0]
        return series[series.index >= start]
    n_days = RANGE_OPTIONS[range_label]
    return series.tail(n_days)


def build_stock_explorer(close, universe=None):
    """
    Interactive dashboard: pick any ticker in the universe and a time
    range, and plot its close price over that window. If `universe`
    (the ticker->sector DataFrame) is provided, the dropdown label
    shows the sector alongside the ticker.
    """
    all_tickers = sorted(close.columns.tolist())

    if universe is not None:
        sector_map = dict(zip(universe["Symbol"], universe["Sector"]))
        display_names = {t: f"{t} ({sector_map.get(t, 'Unknown')})" for t in all_tickers}
    else:
        display_names = {t: t for t in all_tickers}

    name_to_ticker = {v: k for k, v in display_names.items()}

    ticker_dropdown = widgets.Dropdown(
        options=list(display_names.values()),
        value=display_names[all_tickers[0]],
        description="Ticker:",
    )
    range_dropdown = widgets.Dropdown(
        options=list(RANGE_OPTIONS.keys()),
        value="1Y",
        description="Range:",
    )

    controls = widgets.HBox([ticker_dropdown, range_dropdown])
    output = widgets.Output()

    def redraw(change=None):
        output.clear_output(wait=True)
        ticker = name_to_ticker[ticker_dropdown.value]
        series = close[ticker].dropna()
        plot_data = _filter_by_range(series, range_dropdown.value)

        with output:
            if plot_data.empty:
                print(f"No data available for {ticker} in this range.")
                return

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(plot_data.index, plot_data.values, color="black", linewidth=1.8)
            ax.set_title(f"{ticker_dropdown.value} — Close Price ({range_dropdown.value})")
            ax.set_ylabel("Price ($)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

    ticker_dropdown.observe(redraw, names="value")
    range_dropdown.observe(redraw, names="value")

    display(controls, output)
    redraw()

from src.indicators import (
    add_sma, add_ema, add_bollinger_bands, add_rsi, add_macd, add_obv,
    OVERLAY_INDICATORS, SUBPANEL_INDICATORS
)


def build_stock_explorer_with_indicators(close, volume=None, universe=None):
    """
    Interactive dashboard: pick a ticker, an indicator, and a time
    range. Overlay indicators (SMA/EMA/Bollinger) plot directly on the
    price panel; subpanel indicators (Volume/OBV/RSI/MACD) get their
    own row below. Requires `volume` for Volume/OBV to work.
    """
    all_tickers = sorted(close.columns.tolist())

    if universe is not None:
        sector_map = dict(zip(universe["Symbol"], universe["Sector"]))
        display_names = {t: f"{t} ({sector_map.get(t, 'Unknown')})" for t in all_tickers}
    else:
        display_names = {t: t for t in all_tickers}
    name_to_ticker = {v: k for k, v in display_names.items()}

    ticker_dropdown = widgets.Dropdown(
        options=list(display_names.values()),
        value=display_names[all_tickers[0]],
        description="Ticker:",
    )
    indicator_dropdown = widgets.Dropdown(
        options=OVERLAY_INDICATORS + SUBPANEL_INDICATORS,
        value="None",
        description="Indicator:",
    )
    range_dropdown = widgets.Dropdown(
        options=list(RANGE_OPTIONS.keys()),
        value="1Y",
        description="Range:",
    )

    controls = widgets.HBox([ticker_dropdown, indicator_dropdown, range_dropdown])
    output = widgets.Output()

    def redraw(change=None):
        output.clear_output(wait=True)
        ticker = name_to_ticker[ticker_dropdown.value]
        indicator = indicator_dropdown.value
        range_label = range_dropdown.value

        price_series = close[ticker].dropna()
        plot_price = _filter_by_range(price_series, range_label)

        with output:
            if plot_price.empty:
                print(f"No data available for {ticker} in this range.")
                return

            needs_subpanel = indicator in SUBPANEL_INDICATORS

            if needs_subpanel:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(11, 7), sharex=True,
                    gridspec_kw={"height_ratios": [3, 1.3]}
                )
            else:
                fig, ax1 = plt.subplots(figsize=(11, 6))
                ax2 = None

            # --- Price panel ---
            ax1.plot(plot_price.index, plot_price.values, color="black",
                      linewidth=1.8, label="Close")

            if indicator == "SMA 20":
                sma = add_sma(price_series, 20)
                ax1.plot(plot_price.index, sma.loc[plot_price.index],
                          color="blue", linewidth=1, label="SMA 20")
            elif indicator == "SMA 50":
                sma = add_sma(price_series, 50)
                ax1.plot(plot_price.index, sma.loc[plot_price.index],
                          color="orange", linewidth=1, label="SMA 50")
            elif indicator == "EMA 20":
                ema = add_ema(price_series, 20)
                ax1.plot(plot_price.index, ema.loc[plot_price.index],
                          color="green", linewidth=1, label="EMA 20")
            elif indicator == "Bollinger Bands":
                mid, upper, lower = add_bollinger_bands(price_series, 20, 2)
                ax1.plot(plot_price.index, upper.loc[plot_price.index],
                          color="grey", linewidth=1, linestyle="--", label="Upper")
                ax1.plot(plot_price.index, lower.loc[plot_price.index],
                          color="grey", linewidth=1, linestyle="--", label="Lower")
                ax1.fill_between(plot_price.index, lower.loc[plot_price.index],
                                   upper.loc[plot_price.index], color="grey", alpha=0.1)

            ax1.set_title(f"{ticker_dropdown.value} — Close Price"
                          + (f" with {indicator}" if indicator != "None" else "")
                          + f" ({range_label})")
            ax1.set_ylabel("Price ($)")
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(alpha=0.3)

            # --- Sub-panel ---
            if indicator == "Volume":
                if volume is None:
                    ax2.text(0.5, 0.5, "Volume data not provided", ha="center")
                else:
                    vol_series = _filter_by_range(volume[ticker].dropna(), range_label)
                    ax2.bar(vol_series.index, vol_series.values, color="lightblue", width=1.0)
                    ax2.set_ylabel("Volume")

            elif indicator == "OBV":
                if volume is None:
                    ax2.text(0.5, 0.5, "Volume data not provided", ha="center")
                else:
                    obv = add_obv(price_series, volume[ticker])
                    ax2.plot(plot_price.index, obv.loc[plot_price.index],
                              color="purple", linewidth=1.2)
                    ax2.set_ylabel("OBV")

            elif indicator == "RSI":
                rsi = add_rsi(price_series, 14)
                ax2.plot(plot_price.index, rsi.loc[plot_price.index],
                          color="teal", linewidth=1.2)
                ax2.axhline(70, color="red", linestyle="--", linewidth=0.8)
                ax2.axhline(30, color="green", linestyle="--", linewidth=0.8)
                ax2.set_ylabel("RSI")
                ax2.set_ylim(0, 100)

            elif indicator == "MACD":
                macd_line, signal_line, hist = add_macd(price_series)
                ax2.plot(plot_price.index, macd_line.loc[plot_price.index],
                          color="blue", linewidth=1.1, label="MACD")
                ax2.plot(plot_price.index, signal_line.loc[plot_price.index],
                          color="orange", linewidth=1.1, label="Signal")
                ax2.bar(plot_price.index, hist.loc[plot_price.index],
                         color="grey", alpha=0.5, width=1.0)
                ax2.set_ylabel("MACD")
                ax2.legend(loc="upper left", fontsize=8)

            if ax2 is not None:
                ax2.grid(alpha=0.3)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

    ticker_dropdown.observe(redraw, names="value")
    indicator_dropdown.observe(redraw, names="value")
    range_dropdown.observe(redraw, names="value")

    display(controls, output)
    redraw()

def plot_ic_by_sector(sector_ic_df, signal_name, color="steelblue"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(sector_ic_df.index, sector_ic_df["mean_ic"], color=color)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{signal_name} — Mean IC by Sector", fontsize=11)
    ax.set_xlabel("Mean IC", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"ic_by_sector_{signal_name}.png", dpi=150)
    plt.show()

def build_macro_explorer(macro_df, series_labels=None):
    """
    Interactive dashboard: pick any macro series and a time range, and
    plot it over that window. Sparse series (monthly/quarterly data
    like GDP or CPI) are forward-filled for continuous plotting, since
    their native release frequency leaves gaps that would otherwise
    show up as broken lines.
    """
    all_series = sorted(macro_df.columns.tolist())

    if series_labels is not None:
        display_names = {s: f"{s} ({series_labels.get(s, s)})" for s in all_series}
    else:
        display_names = {s: s for s in all_series}
    name_to_series = {v: k for k, v in display_names.items()}

    series_dropdown = widgets.Dropdown(
        options=list(display_names.values()),
        value=display_names[all_series[0]],
        description="Series:",
    )
    range_dropdown = widgets.Dropdown(
        options=list(RANGE_OPTIONS.keys()),
        value="All",
        description="Range:",
    )

    controls = widgets.HBox([series_dropdown, range_dropdown])
    output = widgets.Output()

    def redraw(change=None):
        output.clear_output(wait=True)
        series_code = name_to_series[series_dropdown.value]

        # Forward-fill to handle sparse/lower-frequency series (monthly,
        # quarterly) so the line plots continuously rather than showing
        # gaps between release dates
        series_data = macro_df[series_code].ffill().dropna()
        plot_data = _filter_by_range(series_data, range_dropdown.value)

        with output:
            if plot_data.empty:
                print(f"No data available for {series_code} in this range.")
                return

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(plot_data.index, plot_data.values, color="black", linewidth=1.8)
            ax.set_title(f"{series_dropdown.value} ({range_dropdown.value})")
            ax.set_ylabel("Value")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

    series_dropdown.observe(redraw, names="value")
    range_dropdown.observe(redraw, names="value")

    display(controls, output)
    redraw()

from src.fetch_data import compute_vwap

INTRADAY_RANGE_OPTIONS = {
    "1D": 1,
    "3D": 3,
    "1W": 7,
    "2W": 14,
    "All": None,
}


def build_vwap_explorer(intraday_data):
    """
    Interactive dashboard: pick any ticker with available intraday
    data and a time range (last day, 3 days, week, etc.), and plot
    price alongside its VWAP over that window.
    """
    all_tickers = sorted(intraday_data.keys())

    ticker_dropdown = widgets.Dropdown(
        options=all_tickers,
        value=all_tickers[0],
        description="Ticker:",
    )
    range_dropdown = widgets.Dropdown(
        options=["1D", "3D", "1W", "2W", "1M", "All"],
        value="1W",
        description="Range:",
    )

    controls = widgets.HBox([ticker_dropdown, range_dropdown])
    output = widgets.Output()

    range_to_days = {"1D": 1, "3D": 3, "1W": 7, "2W": 14, "1M": 30, "All": None}

    def redraw(change=None):
        output.clear_output(wait=True)
        ticker = ticker_dropdown.value
        range_label = range_dropdown.value

        df = compute_vwap(intraday_data[ticker])

        n_days = range_to_days[range_label]
        if n_days is not None:
            cutoff = df.index.max() - pd.Timedelta(days=n_days)
            plot_df = df[df.index >= cutoff]
        else:
            plot_df = df

        with output:
            if plot_df.empty:
                print(f"No data available for {ticker} in this range.")
                return

            # Plot against a sequential position index instead of real
            # timestamps, so gaps (nights, weekends) don't draw as flat/
            # diagonal lines. Then manually relabel ticks with real dates.
            x = range(len(plot_df))

            fig, ax = plt.subplots(figsize=(12, 5.5))
            ax.plot(x, plot_df["Close"].values, color="black", linewidth=1.5, label="Price")
            ax.plot(x, plot_df["vwap"].values, color="orange", linewidth=1.3,
                    linestyle="--", label="VWAP")

            # Choose a reasonable number of tick labels (~8) spread across the range
            n_ticks = min(8, len(plot_df))
            tick_positions = [int(i) for i in pd.Series(range(len(plot_df))).quantile(
                [i / (n_ticks - 1) for i in range(n_ticks)]
            )]
            tick_labels = [plot_df.index[pos].strftime("%m/%d %H:%M") for pos in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")

            ax.set_title(f"{ticker} — Price vs. VWAP ({range_label})")
            ax.set_ylabel("Price ($)")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    ticker_dropdown.observe(redraw, names="value")
    range_dropdown.observe(redraw, names="value")

    display(controls, output)
    redraw()