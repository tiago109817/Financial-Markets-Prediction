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

def load_asset(filepath):
    """
    Reads an Excel file and returns a clean DataFrame with:
      - Date       : parsed datetime, sorted from oldest to newest
      - Close      : closing price as float
      - Log_Return : log return computed from Close prices

    Parameters:
        filepath    : path to the .xlsx file
        date_format : (optional) format string for pd.to_datetime
                      Only needed when dates are stored as plain strings (e.g. Bitcoin: "%m/%d/%Y")
                      Leave as None if pandas already reads them as datetime (SP500, Gold, EUR/USD)
    """
    df = pd.read_excel(filepath)

    # Keep only the columns we need
    df = df[["Date", "Price"]].copy()
    df.columns = ["Date", "Close"]

    # Parse dates when the column is a raw string
    df["Date"] = pd.to_datetime(df["Date"])

    # Clean the Close column (removes commas if prices are stored as strings)
    df["Close"] = clean_price(df["Close"])

    # Sort from oldest to newest so log returns are computed in the right order
    df = df.sort_values("Date").reset_index(drop=True)

    # Compute log returns
    df["Log_Return"] = compute_log_returns(df["Close"])
    
    # Compute moving average
    df["Moving_Average"] = compute_moving_average(df["Close"])
    
    return df


# ─────────────────────────────────────────────
# Load all assets
# ─────────────────────────────────────────────

# SP500, Gold and EUR/USD: pandas reads the Date column as datetime automatically
# Bitcoin: dates are plain strings in MM/DD/YYYY format, so we pass the format explicitly

sp500   = load_asset("./data/SP500_2000_2025.xlsx")
gold    = load_asset("./data/Gold_2000_2025.xlsx")
eur_usd = load_asset("./data/EUR_USD_2000_2025.xlsx")
bitcoin = load_asset("./data/Bitcoin_2010_2025.xlsx")