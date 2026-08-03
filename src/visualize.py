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