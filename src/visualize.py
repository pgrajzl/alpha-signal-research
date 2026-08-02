"""
visualize.py
Plots IC time series, decay curves, and cumulative IC for signal
evaluation.
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


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