import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ─────────────────────────────────────────────
# Shared plot settings
# ─────────────────────────────────────────────

# Consistent color mapping for each asset across all visualizations
# This improves readability and comparability between charts

COLORS = {
    "SP500":   "#F33621",   # red
    "Gold":    "#FFC107",   # amber
    "EUR/USD": "#4CAF50",   # green
    "Bitcoin": "#22BDFF",   # cyan
}

# Mapping between visualization modes and DataFrame columns
# Allows reuse of the same plotting logic for different representations

CHART_MODES = {
    "close": ("Close",          "Close Price over Time"),
    "ma":    ("Moving_Average", "Moving Average over Time"),
}

# Standardized x-axis formatting for time series plots
# - Major ticks: yearly labels
# - Minor ticks: quarterly markers (every 3 months)
# - Vertical grid lines improve temporal alignment and interpretation

def _format_x_axis(ax):
    # Major ticks: yearly (labeled)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Minor ticks: EXACT quarter starts (Jan, Apr, Jul, Oct 1st)
    ax.xaxis.set_minor_locator(
        mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1)
    )

    # Vertical grid lines at those exact dates
    ax.grid(which='both', axis='x', linestyle='--', alpha=0.3)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")



# ─────────────────────────────────────────────
# Main plot function
# ─────────────────────────────────────────────

def plot_assets(assets, mode):
    """
    Plot each asset in a separate figure.

    Parameters:
        assets : dict
            Mapping {asset_name: DataFrame}
        mode : str
            "close" → raw prices
            "ma"    → moving average (smoothed series)

    Each chart displays a single time series to allow clear
    visual inspection without overlap.
    """
    if mode not in CHART_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(CHART_MODES.keys())}")

    # Select column and title based on chosen mode
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
    Plot log returns for all assets on a single chart.

    This representation highlights:
    - Volatility clustering
    - Relative variability across assets
    - Mean-reverting behavior around zero

    Using a shared axis facilitates direct comparison.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Log Returns over Time", fontsize=16, fontweight="bold")

    for name, df in assets.items():
        ax.plot(
            df["Date"],
            df["Log_Return"],
            color=COLORS[name],
            linewidth=0.6,
            alpha=0.8,
            label=name
        )

    # Zero line: reference for positive/negative returns
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_ylabel("Log Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _format_x_axis(ax)

    plt.tight_layout()
    plt.show()
    

# ─────────────────────────────────────────────
# ARIMA Evaluation (Quarterly prints)
# ─────────────────────────────────────────────

def print_quarterly_results(name, forecast_df):
    """
    Print forecast vs real values at fixed quarterly checkpoints.

    Dates evaluated:
        - April 1
        - July 1
        - October 1
        - January 1 (following year)

    This provides a standardized way to compare model performance
    at consistent horizons throughout the forecast period.
    """

    print(f"\n{name} — Quarterly Forecast Evaluation")
    print("=" * 60)

    target_dates = [
        f"{forecast_df['Date'].dt.year.iloc[0]}-04-01",
        f"{forecast_df['Date'].dt.year.iloc[0]}-07-01",
        f"{forecast_df['Date'].dt.year.iloc[0]}-10-01",
        f"{forecast_df['Date'].dt.year.iloc[0] + 1}-01-01",
    ]

    for d in target_dates:
        d = pd.Timestamp(d)

        row = forecast_df[forecast_df["Date"] == d]

        if not row.empty:
            f = row["Forecast"].values[0]
            r = row["Real"].values[0]

            # Relative error (%)
            error = ((f - r) / r) * 100

            # Higher precision for exchange rates
            if name == "EUR/USD":
                print(f"{d.date()} | Forecast: {f:.4f} | Real: {r:.4f} | Error: {error:.2f}%")
            else:
                print(f"{d.date()} | Forecast: {f:.2f} | Real: {r:.2f} | Error: {error:.2f}%")

        else:
            print(f"{d.date()} | No data available")


# ─────────────────────────────────────────────
# ARIMA Plot
# ─────────────────────────────────────────────

def plot_forecast_with_train(name, train_df, forecast_df):
    """
    Plot ARIMA forecasting results.

    Components:
        - Training data (historical fit period)
        - Real observed values (out-of-sample)
        - Forecasted values

    This visualization allows direct assessment of:
        - Forecast accuracy
        - Deviation over time
        - Structural differences between prediction and reality
    """

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"{name}", fontsize=14, fontweight="bold")
    
    # Historical training data
    ax.plot(
        train_df["Date"],
        train_df["Close"],
        label="Train",
        linewidth=1.2
    )

    # Real observed future values
    ax.plot(
        forecast_df["Date"],
        forecast_df["Real"],
        label="Real",
        linewidth=1.5
    )

    # Model forecast (dashed for visual distinction)
    ax.plot(
        forecast_df["Date"],
        forecast_df["Forecast"],
        linestyle="--",
        label="Forecast",
        linewidth=1.5
    )
    
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(axis='y', alpha=0.5)

    _format_x_axis(ax)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Random Walk Plot  (expected path + sigma bands)
# ─────────────────────────────────────────────

def plot_rw_forecast(name, train_df, forecast_df):
    """
    Plot Random Walk forecasting results with GBM confidence bands.

    Components:
        - Training data (historical fit period)
        - Real observed values (out-of-sample)
        - Expected path  (deterministic drift, r_t = mu for all t)
        - ±1σ band  (shaded, darker)  — ~68% of random paths fall inside
        - ±2σ band  (shaded, lighter) — ~95% of random paths fall inside

    The bands are derived from the GBM distributional result:

        log(P_t / P_0) ~ Normal(mu * t,  sigma² * t)

    so the band width grows proportionally to sqrt(t) from each anchor.
    For the monthly model the bands restart from the new anchor every month;
    for the static and long-run models they expand continuously from day 1.

    Parameters
    ----------
    name        : str         Title label (asset name + model description).
    train_df    : DataFrame   Historical data with columns Date, Close.
    forecast_df : DataFrame   Output from any forecast_rw_* function.
                              Must contain: Date, Forecast, Real,
                              Upper1, Lower1, Upper2, Lower2.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"{name}", fontsize=14, fontweight="bold")

    dates = forecast_df["Date"]

    # Reuse the same matplotlib default colors as plot_forecast_with_train:
    #   C0 (blue)   → Train
    #   C1 (orange) → Real
    #   C2 (green)  → Forecast / Expected path
    # Bands are tinted C2 so they read as an extension of the forecast line.

    # ── ±2σ band (wider, lightest) ────────────────────────────────────────────
    ax.fill_between(
        dates,
        forecast_df["Lower2"],
        forecast_df["Upper2"],
        alpha=0.12,
        color="C2",
        label="±2σ band  (~95%)",
    )

    # ── ±1σ band (narrower, slightly darker) ─────────────────────────────────
    ax.fill_between(
        dates,
        forecast_df["Lower1"],
        forecast_df["Upper1"],
        alpha=0.25,
        color="C2",
        label="±1σ band  (~68%)",
    )

    # ── Training data (C0 blue — identical to ARIMA plot) ────────────────────
    ax.plot(
        train_df["Date"],
        train_df["Close"],
        color="C0",
        linewidth=1.2,
        label="Train",
    )

    # ── Real observed values (C1 orange — identical to ARIMA plot) ───────────
    ax.plot(
        dates,
        forecast_df["Real"],
        color="C1",
        linewidth=1.5,
        label="Real",
    )

    # ── Expected path (C2 green, solid + thicker — stands out over the bands) ─
    ax.plot(
        dates,
        forecast_df["Forecast"],
        color="C2",
        linestyle="--",
        linewidth=2.0,
        label="Expected path  (μ drift)",
    )

    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.5)

    _format_x_axis(ax)

    plt.tight_layout()
    plt.show()