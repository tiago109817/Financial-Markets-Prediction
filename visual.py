import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─────────────────────────────────────────────
# Shared plot settings
# ─────────────────────────────────────────────

# Colors for each asset (used consistently across all charts)

COLORS = {
    "SP500":   "#F33621",   # red
    "Gold":    "#FFC107",   # amber
    "EUR/USD": "#4CAF50",   # green
    "Bitcoin": "#22BDFF",   # cyan
}

# Each mode corresponds to a different column and title suffix

CHART_MODES = {
    "close": ("Close",          "Close Price over Time"),
    "ma":    ("Moving_Average", "Moving Average over Time"),
}

# Default date format for an yearly x-axis (used in all charts) with vertical lines every 3 months
def _format_x_axis(ax):
    # Major ticks: yearly (with labels)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Minor ticks: every 3 months (no labels)
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

    # Draw vertical grid lines for minor ticks
    ax.grid(which='minor', axis='x', linestyle='--', alpha=0.3)

    # Keep your label formatting
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")



# ─────────────────────────────────────────────
# Main plot function
# ─────────────────────────────────────────────

def plot_assets(assets, mode):
    """
    Plots a chart for each asset in its own figure.

    Parameters:
        assets : dict   {name: DataFrame}  e.g. {"SP500": sp500, ...}
        mode   : str    "close" or "ma"
    """
    if mode not in CHART_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(CHART_MODES.keys())}")

    # Column to plot and title suffix based on the mode
    col, title_suffix = CHART_MODES[mode]

    for name, df in assets.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.suptitle(f"{name} — {title_suffix}", fontsize=14, fontweight="bold")
        ax.plot(df["Date"], df[col], color=COLORS[name], linewidth=1.2)
        ax.set_ylabel("Price")
        ax.grid(axis='y', alpha=0.5)
        _format_x_axis(ax)
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
# Log Returns — all assets on a single chart
# ─────────────────────────────────────────────

def plot_log_returns(assets):
    """
    Plots Log Returns vs Time for all assets on the same chart.

    Parameters:
        assets : dict  {name: DataFrame}  e.g. {"SP500": sp500, ...}
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Log Returns over Time", fontsize=16, fontweight="bold")

    for name, df in assets.items():
        ax.plot(df["Date"], df["Log_Return"], color=COLORS[name],
                linewidth=0.6, alpha=0.8, label=name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Log Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _format_x_axis(ax)

    plt.tight_layout()
    plt.show()