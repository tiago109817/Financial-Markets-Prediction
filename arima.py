import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


# ─────────────────────────────────────────────
# Helper: fit ARIMA on log returns
# ─────────────────────────────────────────────

def fit_arima(log_returns, dates, order=(2, 0, 2)):
    """
    Fit an ARIMA(p,d,q) model to a log-return series.

    The input is explicitly converted into a pandas Series with a
    DatetimeIndex to ensure correct temporal structure.

    A daily frequency is enforced to guarantee equally spaced observations,
    which is a key assumption for ARIMA estimation.
    """

    series = pd.Series(
        log_returns.values,
        index=pd.DatetimeIndex(dates)
    ).dropna()

    # Enforce regular daily spacing (required for consistency)
    series = series.asfreq("D")

    model = ARIMA(series, order=order)
    fitted = model.fit()

    return fitted


# ─────────────────────────────────────────────
# Helper: reconstruct prices from log returns
# ─────────────────────────────────────────────

def reconstruct_prices(last_price, forecasted_returns):
    """
    Reconstruct a price path from forecasted log returns.

    Given:
        r_t = ln(P_t / P_{t-1})

    The price evolves as:
        P_t = P_{t-1} * exp(r_t)

    The function iteratively applies this relation starting from
    the last observed price.
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
    Display key ARIMA diagnostics.

    Includes:
    - Model order (p, d, q)
    - Information criteria (AIC, BIC) for model comparison
    - Estimated parameters

    Useful for evaluating model fit and complexity.
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
    Static (one-shot) ARIMA forecast.

    Procedure:
    - Train model using data up to Jan 1st of year_n
    - Forecast log returns for the full following year (daily horizon)
    - Convert forecasted returns into price levels

    Returns:
        train       : training dataset (for visualization/analysis)
        forecast_df : DataFrame with Date, Forecast, and Real prices
    """

    # ─────────────────────────────────────────
    # Train / test split
    # ─────────────────────────────────────────

    cutoff = pd.Timestamp(f"{year_n}-01-01")
    end    = pd.Timestamp(f"{year_n+1}-01-01")

    train = df[df["Date"] < cutoff].copy()
    test  = df[(df["Date"] >= cutoff) & (df["Date"] <= end)].copy()

    # ─────────────────────────────────────────
    # Model estimation
    # ─────────────────────────────────────────

    model = fit_arima(train["Log_Return"], train["Date"], order=order)
    print_model_summary(name, model)

    # ─────────────────────────────────────────
    # Forecast log returns
    # ─────────────────────────────────────────

    steps = len(test)
    forecast_returns = model.forecast(steps=steps)

    # ─────────────────────────────────────────
    # Transform returns → prices
    # ─────────────────────────────────────────

    last_price = train["Close"].iloc[-1]
    forecast_prices = reconstruct_prices(last_price, forecast_returns)

    # ─────────────────────────────────────────
    # Assemble results
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
    Rolling (recursive) ARIMA forecasting with periodic retraining.

    Procedure:
    - Initial training up to Jan 1st of year_n
    - Forecast one month ahead
    - Incorporate REAL observed data into training set
    - Retrain model and repeat

    This mimics a realistic setting where models are updated as new
    information becomes available.

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

        # Define next monthly boundary
        next_month = (current_date + pd.offsets.MonthBegin(1))

        # Extract current month's observations
        test_slice = test[
            (test["Date"] >= current_date) &
            (test["Date"] < next_month)
        ].copy()

        if test_slice.empty:
            current_date = next_month
            continue

        # ─────────────────────────────────────
        # Fit model on updated training data
        # ─────────────────────────────────────

        model = fit_arima(
            current_train["Log_Return"],
            current_train["Date"],
            order=order
        )

        print_model_summary(name, model)

        # ─────────────────────────────────────
        # Forecast current month
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

        # Advance to next month
        current_date = next_month

    # Combine all monthly forecasts
    forecast_df = pd.concat(all_forecasts, ignore_index=True)

    return train, forecast_df


# ─────────────────────────────────────────────
# Monthly rolling (self-propagating forecast)
# ─────────────────────────────────────────────

def forecast_asset_monthly_propagating(name, df, year_n, order=(2, 0, 2)):
    """
    Rolling ARIMA with self-propagating price dynamics.

    Same structure as the standard rolling approach, with one key difference:
    - The forecast path is continuous across months
    - Each new forecast starts from the PREVIOUS FORECASTED price,
      not the last observed real price

    Implication:
    - Errors accumulate over time (more realistic for long-horizon evaluation)
    - Avoids artificial “resets” to true values at each retraining step
    """

    cutoff = pd.Timestamp(f"{year_n}-01-01")
    end    = pd.Timestamp(f"{year_n+1}-01-01")

    train = df[df["Date"] < cutoff].copy()
    test  = df[(df["Date"] >= cutoff) & (df["Date"] <= end)].copy()

    all_forecasts = []

    current_train = train.copy()
    current_date = cutoff

    # Initial price anchor (last observed real value)
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

        # Forecast returns
        steps = len(test_slice)
        forecast_returns = model.forecast(steps=steps)

        # Use last forecasted price (not real price)
        forecast_prices = reconstruct_prices(last_price, forecast_returns)

        # Update anchor with final forecasted value
        last_price = forecast_prices[-1]

        forecast_df = pd.DataFrame({
            "Date": test_slice["Date"].values,
            "Forecast": forecast_prices,
            "Real": test_slice["Close"].values
        })

        all_forecasts.append(forecast_df)

        # Model still learns from REAL observations
        current_train = pd.concat([current_train, test_slice], ignore_index=True)

        current_date = next_month

    forecast_df = pd.concat(all_forecasts, ignore_index=True)

    return train, forecast_df
