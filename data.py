import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def clean_price(series):
    """
    Convert a price column to numeric (float).

    Some datasets store prices as strings with thousands separators
    (e.g., "1,455.20"), which must be removed before conversion.
    """
    return series.astype(str).str.replace(",", "").astype(float)


def compute_log_returns(prices):
    """
    Compute logarithmic returns of a price series.

    Formula:
        r_t = ln(P_t / P_{t-1})

    Log returns will be useful as they are stationary and close to zero. 
    The first observation is NaN.
    """
    return np.log(prices / prices.shift(1))


def compute_moving_average(prices):
    """
    Compute a centered moving average (window size = 13).

    This smooths short-term fluctuations in the price series by averaging
    neighboring values symmetrically around each point.

    Boundary values are NaN due to insufficient surrounding data.
    
    This was done as I thought the original data had a lot of noise and 
    I wanted to have a smoother series for visualization and ARIMA modeling.
    
    I then discovered a misformatted date column in the original Excel files, 
    which caused the noise. After fixing the date parsing, the data is much cleaner, 
    but I kept the moving average as an optional feature.
    """
    return (
        prices + prices.shift(1) + prices.shift(-1)
        + prices.shift(2) + prices.shift(-2)
        + prices.shift(3) + prices.shift(-3)
        + prices.shift(4) + prices.shift(-4)
        + prices.shift(5) + prices.shift(-5)
        + prices.shift(6) + prices.shift(-6)
    ) / 13


def fill_missing_days(df):
    """
    Ensure a continuous daily time index.

    - Expands the dataset to include all calendar days
    - Fills missing prices (e.g., weekends, holidays) using forward fill
    - Recomputes derived variables on the completed series

    This step is required for time-series models (e.g., ARIMA) that assume
    regularly spaced observations.
    """
    df = df.set_index("Date")

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_index)

    # Forward-fill missing prices
    df["Close"] = df["Close"].ffill()

    # Recompute derived series
    df["Log_Return"]     = compute_log_returns(df["Close"])
    df["Moving_Average"] = compute_moving_average(df["Close"])

    df.index.name = "Date"
    df = df.reset_index()

    return df

# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def load_asset(filepath):
    """
    Load and preprocess a financial time series from Excel.

    Output DataFrame:
        - Date           : daily datetime index (no gaps)
        - Close          : cleaned closing price (float)
        - Log_Return     : logarithmic returns
        - Moving_Average : smoothed price series

    Expected Excel format:
        Month | Day | Year | Close
    """
    df = pd.read_excel(filepath)

    # Construct a proper datetime column (avoids ambiguity)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" +
        df["Month"].astype(int).astype(str).str.zfill(2) + "-" +
        df["Day"].astype(int).astype(str).str.zfill(2)
    )

    df = df[["Date", "Close"]].copy()

    # Clean and standardize price data
    df["Close"] = clean_price(df["Close"])

    # Ensure chronological ordering
    df = df.sort_values("Date").reset_index(drop=True)

    # Fill missing dates and compute derived features
    df = fill_missing_days(df)

    return df


# ─────────────────────────────────────────────
# Load all assets
# ─────────────────────────────────────────────

sp500   = load_asset("./data/SP500_2000_2025.xlsx")
gold    = load_asset("./data/Gold_2000_2025.xlsx")
eur_usd = load_asset("./data/EUR_USD_2000_2025.xlsx")
bitcoin = load_asset("./data/Bitcoin_2010_2025.xlsx")