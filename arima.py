from xml.parsers.expat import model

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


# ─────────────────────────────────────────────
# Helper: fit ARIMA on log returns
# ─────────────────────────────────────────────

def fit_arima(log_returns, dates, order=(2, 0, 2)):
    """
    Fit ARIMA(p,d,q) on log returns with proper datetime handling.
    """

    series = pd.Series(
        log_returns.values,
        index=pd.DatetimeIndex(dates)
    ).dropna()

    # Ensure regular spacing (critical for ARIMA)
    series = series.asfreq("D")

    model = ARIMA(
        series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(
        method_kwargs={
            "maxiter": 300
        }
    )


    return fitted


# ─────────────────────────────────────────────
# Helper: reconstruct prices
# ─────────────────────────────────────────────

def reconstruct_prices(last_price, forecasted_returns):
    """
    Convert log returns → price path.
    """

    prices = [last_price]

    for r in forecasted_returns:
        prices.append(prices[-1] * np.exp(r))

    return prices[1:]


# ─────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────

def print_model_summary(name, model):
    """
    Print ARIMA diagnostics (optional).
    """

    print(f"\n{name} — ARIMA Model Summary")
    print("=" * 60)

    print(f"Order (p,d,q): {model.model_orders}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    print("\nParameters:")
    print(model.params)


# ─────────────────────────────────────────────
# CORE ENGINE (IMPORTANT)
# ─────────────────────────────────────────────

def _run_forecast(df, start, end, order,
                  expanding=True,
                  window=None,
                  propagating=False):
    """
    ⚙️ Generic ARIMA forecasting engine.

    This function powers ALL dynamic models.

    PARAMETERS
    ----------
    expanding   : True  → training grows over time
                  False → fixed rolling window

    window      : number of months (only used if expanding=False)

    propagating : False → reset to REAL price each step
                  True  → continue from FORECAST (drift allowed)

    RETURNS
    -------
    forecast_df : DataFrame with Date, Forecast, Real

    ─────────────────────────────────────────

    MODEL TYPES (based on parameters)

    Expanding + Real       → Monthly retraining
    Expanding + Forecast   → Monthly propagating
    Rolling   + Real       → Rolling window
    Rolling   + Forecast   → Rolling propagating
    """

    all_forecasts = []
    current_date = start

    # Initial anchor (used only for propagating models)
    last_price = df[df["Date"] < start]["Close"].iloc[-1]

    # Initial expanding training set
    if expanding:
        current_train = df[df["Date"] < start].copy()

    while current_date <= end:

        next_month = current_date + pd.offsets.MonthBegin(1)
        if next_month > end:
            next_month = end + pd.Timedelta(days=1)  # ensure we include the end date

        # ─────────────────────────────────────
        # Select training data
        # ─────────────────────────────────────

        if expanding:
            train_slice = current_train.copy()
        else:
            train_start = current_date - pd.DateOffset(months=window)
            train_slice = df[
                (df["Date"] >= train_start) &
                (df["Date"] < current_date)
            ].copy()

        if len(train_slice) < 50:
            current_date = next_month
            continue

        # ─────────────────────────────────────
        # Select test data (this month)
        # ─────────────────────────────────────

        test_slice = df[
            (df["Date"] >= current_date) &
            (df["Date"] < next_month)
        ].copy()

        if test_slice.empty:
            current_date = next_month
            continue

        # ─────────────────────────────────────
        # Fit model
        # ─────────────────────────────────────

        model = fit_arima(
            train_slice["Log_Return"],
            train_slice["Date"],
            order=order
        )

        # ─────────────────────────────────────
        # Forecast
        # ─────────────────────────────────────

        steps = len(test_slice)
        forecast_returns = model.forecast(steps=steps)

        # Anchor logic
        if propagating:
            base_price = last_price
        else:
            base_price = train_slice["Close"].iloc[-1]

        forecast_prices = reconstruct_prices(base_price, forecast_returns)

        # Update anchor if propagating
        if propagating:
            last_price = forecast_prices[-1]

        forecast_df = pd.DataFrame({
            "Date": test_slice["Date"].values,
            "Forecast": forecast_prices,
            "Real": test_slice["Close"].values
        })

        all_forecasts.append(forecast_df)

        # Update expanding training
        if expanding:
            current_train = pd.concat([current_train, test_slice], ignore_index=True)

        current_date = next_month

    return pd.concat(all_forecasts, ignore_index=True)


# ─────────────────────────────────────────────
# 1. STATIC MODEL (no engine used)
# ─────────────────────────────────────────────

def forecast_asset(name, df, year_n, order=(2, 0, 2)):

    cutoff = pd.Timestamp(f"{year_n}-01-01")
    end    = pd.Timestamp(f"{year_n+1}-01-01")

    train = df[df["Date"] < cutoff].copy()
    test  = df[(df["Date"] >= cutoff) & (df["Date"] <= end)].copy()

    model = fit_arima(train["Log_Return"], train["Date"], order=order)
    print_model_summary(name, model)

    steps = len(test)
    forecast_returns = model.forecast(steps=steps)

    last_price = train["Close"].iloc[-1]
    forecast_prices = reconstruct_prices(last_price, forecast_returns)

    forecast_df = pd.DataFrame({
        "Date": test["Date"].values,
        "Forecast": forecast_prices,
        "Real": test["Close"].values
    })

    return train, forecast_df


# ─────────────────────────────────────────────
# 2. MONTHLY (expanding, real anchor)
# ─────────────────────────────────────────────

def forecast_asset_monthly(name, df, year_n, order=(2, 0, 2)):

    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n+1}-01-01")

    train = df[df["Date"] < start].copy()

    forecast_df = _run_forecast(
        df, start, end,
        order=order,
        expanding=True,
        propagating=False
    )

    return train, forecast_df


# ─────────────────────────────────────────────
# 3. MONTHLY PROPAGATING
# ─────────────────────────────────────────────

def forecast_asset_monthly_propagating(name, df, year_n, order=(2, 0, 2)):

    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n+1}-01-01")

    train = df[df["Date"] < start].copy()

    forecast_df = _run_forecast(
        df, start, end,
        order=order,
        expanding=True,
        propagating=True
    )

    return train, forecast_df


# ─────────────────────────────────────────────
# 4. ROLLING WINDOW (12 months)
# ─────────────────────────────────────────────

def forecast_asset_rolling_window(name, df, year_n, order=(2, 0, 2)):

    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n+1}-01-01")

    forecast_df = _run_forecast(
        df, start, end,
        order=order,
        expanding=False,
        window=24,
        propagating=False
    )

    df_plot = df[df["Date"] <= end].copy()

    return df_plot, forecast_df


# ─────────────────────────────────────────────
# 5. ROLLING WINDOW PROPAGATING
# ─────────────────────────────────────────────

def forecast_asset_rolling_window_propagating(name, df, year_n, order=(2, 0, 2)):

    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n+1}-01-01")

    forecast_df = _run_forecast(
        df, start, end,
        order=order,
        expanding=False,
        window=24,
        propagating=True
    )

    df_plot = df[df["Date"] <= end].copy()

    return df_plot, forecast_df
