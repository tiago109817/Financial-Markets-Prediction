import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def clean_price(series):
    """
    Converts a price column to float.
    Needed because some files store prices as strings with commas (e.g. "1,455.20").
    """
    return series.astype(str).str.replace(",", "").astype(float)


def compute_log_returns(prices):
    """
    Computes log returns from a price series.
    Log return at time t = ln(P_t / P_{t-1})
    The first value is NaN because there is no previous price.
    """
    return np.log(prices / prices.shift(1))


def compute_moving_average(prices):
    """
    Computes moving average from a price series.
    This is done to smooth out the price vs time curve.
    Some of the first and last values are NaN because there is no previous or next price.
    """
    return (prices + prices.shift(1) + prices.shift(-1) + prices.shift(2) + prices.shift(-2) + prices.shift(3)
            + prices.shift(-3) + prices.shift(4) + prices.shift(-4) + prices.shift(5) + prices.shift(-5) + prices.shift(6) + prices.shift(-6)) / 13


def fill_missing_days(df):
    """
    Expands a DataFrame to cover every calendar day between its first and last date.
    Days that were missing (weekends, holidays) are forward-filled from the most
    recent known closing price, so the series has no gaps.

    This is required for ARIMA and other time-series models that assume a
    regularly-spaced, gap-free index.
    """
    df = df.set_index("Date")

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_index)

    df["Close"] = df["Close"].ffill()

    df["Log_Return"]     = compute_log_returns(df["Close"])
    df["Moving_Average"] = compute_moving_average(df["Close"])

    df.index.name = "Date"
    df = df.reset_index()

    return df


def load_asset(filepath):
    """
    Reads an Excel file and returns a clean DataFrame with:
      - Date           : parsed datetime, one row per calendar day (no gaps)
      - Close          : closing price as float (weekends/holidays forward-filled)
      - Log_Return     : log return computed from the filled Close series
      - Moving_Average : moving average computed from the filled Close series

    Excel format expected:
        Month | Day | Year | Close
    """
    df = pd.read_excel(filepath)

    # Build Date from the three separate columns — zero ambiguity
    df["Date"] = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" +
        df["Month"].astype(int).astype(str).str.zfill(2) + "-" +
        df["Day"].astype(int).astype(str).str.zfill(2)
    )

    df = df[["Date", "Close"]].copy()

    df["Close"] = clean_price(df["Close"])

    df = df.sort_values("Date").reset_index(drop=True)

    df = fill_missing_days(df)

    return df


# ─────────────────────────────────────────────
# Load all assets
# ─────────────────────────────────────────────

sp500   = load_asset("./data/SP500_2000_2025.xlsx")
gold    = load_asset("./data/Gold_2000_2025.xlsx")
eur_usd = load_asset("./data/EUR_USD_2000_2025.xlsx")
bitcoin = load_asset("./data/Bitcoin_2010_2025.xlsx")