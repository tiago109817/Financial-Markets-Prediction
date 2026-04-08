import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


# ─────────────────────────────────────────────
# Helper: fit ARIMA on log returns
# ─────────────────────────────────────────────

def fit_arima(log_returns, dates, order=(2, 0, 2)):
    """
    Fits ARIMA using a proper DatetimeIndex.
    """

    series = pd.Series(
        log_returns.values,
        index=pd.DatetimeIndex(dates)
    ).dropna()

    # enforce daily frequency (CRITICAL)
    series = series.asfreq("D")

    model = ARIMA(series, order=order)
    fitted = model.fit()

    return fitted


# ─────────────────────────────────────────────
# Helper: reconstruct prices from log returns
# ─────────────────────────────────────────────

def reconstruct_prices(last_price, forecasted_returns):
    """
    Converts forecasted log returns into price series.
    """
    prices = [last_price]

    for r in forecasted_returns:
        next_price = prices[-1] * np.exp(r)
        prices.append(next_price)

    return prices[1:]

# ─────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────

def print_model_summary(name, model):
    """
    Prints ARIMA model parameters and diagnostics.
    """

    print(f"\n{name} — ARIMA Model Summary")
    print("=" * 60)

    print(f"Order (p,d,q): {model.model_orders}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    print("\nParameters:")
    print(model.params)


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def forecast_asset(name, df, year_n, order=(2, 0, 2)):
    """
    Trains ARIMA until Jan 1st of year_n and forecasts 1 year ahead (daily).

    Returns:
        forecast_df : DataFrame with Date, Forecast, Real
        train       : DataFrame with training data (for plotting)
    """

    # ─────────────────────────────────────────
    # Split train / test
    # ─────────────────────────────────────────

    cutoff = pd.Timestamp(f"{year_n}-01-01")
    end    = pd.Timestamp(f"{year_n+1}-01-01")

    train = df[df["Date"] < cutoff].copy()
    test  = df[(df["Date"] >= cutoff) & (df["Date"] <= end)].copy()

    # ─────────────────────────────────────────
    # Fit model
    # ─────────────────────────────────────────

    model = fit_arima(train["Log_Return"], train["Date"], order=order)
    print_model_summary(name, model)

    # ─────────────────────────────────────────
    # Forecast log returns
    # ─────────────────────────────────────────

    steps = len(test)
    forecast_returns = model.forecast(steps=steps)

    # ─────────────────────────────────────────
    # Convert to prices
    # ─────────────────────────────────────────

    last_price = train["Close"].iloc[-1]
    forecast_prices = reconstruct_prices(last_price, forecast_returns)

    # ─────────────────────────────────────────
    # Build result DataFrame
    # ─────────────────────────────────────────

    forecast_df = pd.DataFrame({
        "Date": test["Date"].values,
        "Forecast": forecast_prices,
        "Real": test["Close"].values
    })

    return train, forecast_df


# ─────────────────────────────────────────────
# Monthly rolling forecast (retraining)
# ─────────────────────────────────────────────

def forecast_asset_monthly(name, df, year_n, order=(2, 0, 2)):
    """
    Rolling ARIMA:
    - Train until Jan 1st of year_n
    - Forecast month by month
    - Retrain after each month using real data

    Returns:
        train_df, forecast_df
    """

    cutoff = pd.Timestamp(f"{year_n}-01-01")
    end    = pd.Timestamp(f"{year_n+1}-01-01")

    train = df[df["Date"] < cutoff].copy()
    test  = df[(df["Date"] >= cutoff) & (df["Date"] <= end)].copy()

    all_forecasts = []

    current_train = train.copy()

    current_date = cutoff

    while current_date <= end:

        # Define next month boundary
        next_month = (current_date + pd.offsets.MonthBegin(1))

        # Slice this month's test data
        test_slice = test[
            (test["Date"] >= current_date) &
            (test["Date"] < next_month)
        ].copy()

        if test_slice.empty:
            current_date = next_month
            continue

        # ─────────────────────────────────────
        # Fit model on current training data
        # ─────────────────────────────────────

        model = fit_arima(
            current_train["Log_Return"],
            current_train["Date"],
            order=order
        )

        print_model_summary(name, model)

        # ─────────────────────────────────────
        # Forecast this month
        # ─────────────────────────────────────

        steps = len(test_slice)
        forecast_returns = model.forecast(steps=steps)

        last_price = current_train["Close"].iloc[-1]
        forecast_prices = reconstruct_prices(last_price, forecast_returns)

        forecast_df = pd.DataFrame({
            "Date": test_slice["Date"].values,
            "Forecast": forecast_prices,
            "Real": test_slice["Close"].values
        })

        all_forecasts.append(forecast_df)

        # ─────────────────────────────────────
        # Update training set with REAL data
        # ─────────────────────────────────────

        current_train = pd.concat([current_train, test_slice], ignore_index=True)

        # Move to next month
        current_date = next_month

    # Combine all months
    forecast_df = pd.concat(all_forecasts, ignore_index=True)

    return train, forecast_df


# ─────────────────────────────────────────────
# Monthly rolling (self-propagating forecast)
# ─────────────────────────────────────────────

def forecast_asset_monthly_propagating(name, df, year_n, order=(2, 0, 2)):
    """
    Rolling ARIMA:
    - Train until Jan 1st of year_n
    - Forecast month by month
    - Retrain with real data BUT continue price from previous forecast

    Key difference:
    Forecast path is continuous (does NOT reset to real price each month)
    """

    cutoff = pd.Timestamp(f"{year_n}-01-01")
    end    = pd.Timestamp(f"{year_n+1}-01-01")

    train = df[df["Date"] < cutoff].copy()
    test  = df[(df["Date"] >= cutoff) & (df["Date"] <= end)].copy()

    all_forecasts = []

    current_train = train.copy()
    current_date = cutoff

    # THIS is the key difference - restarts from the last forecast, not from last data
    last_price = train["Close"].iloc[-1]

    while current_date <= end:

        next_month = current_date + pd.offsets.MonthBegin(1)

        test_slice = test[
            (test["Date"] >= current_date) &
            (test["Date"] < next_month)
        ].copy()

        if test_slice.empty:
            current_date = next_month
            continue

        # Fit model
        model = fit_arima(
            current_train["Log_Return"],
            current_train["Date"],
            order=order
        )

        # Forecast
        steps = len(test_slice)
        forecast_returns = model.forecast(steps=steps)

        # KEY CHANGE: use rolling forecast price
        forecast_prices = reconstruct_prices(last_price, forecast_returns)

        # Update last_price to LAST FORECAST (not real)
        last_price = forecast_prices[-1]

        forecast_df = pd.DataFrame({
            "Date": test_slice["Date"].values,
            "Forecast": forecast_prices,
            "Real": test_slice["Close"].values
        })

        all_forecasts.append(forecast_df)

        # Update training with REAL data (model still learns reality)
        current_train = pd.concat([current_train, test_slice], ignore_index=True)

        current_date = next_month

    forecast_df = pd.concat(all_forecasts, ignore_index=True)

    return train, forecast_df
