import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_params(log_returns):
    """
    Estimate the two parameters of the random walk from a log-return series.

    A Geometric Brownian Motion (GBM) random walk is fully characterised by:

      mu    (drift / expected daily log return)
          = sample mean of log returns

      sigma (volatility / standard deviation of daily log returns)
          = sample standard deviation of log returns

    The forecast uses the *expected path* (deterministic drift, r_t = mu for
    all t) rather than random draws, so results are reproducible and directly
    comparable with ARIMA point forecasts.

    Parameters
    ----------
    log_returns : array-like   Series of log returns (NaN values dropped).

    Returns
    -------
    mu    : float   Mean daily log return.
    sigma : float   Standard deviation of daily log returns.
    """
    clean = log_returns.dropna()
    mu    = clean.mean()
    sigma = clean.std(ddof=1)
    return mu, sigma


def _reconstruct_prices(anchor, log_return_forecasts):
    """
    Convert a sequence of forecasted log returns into a price path.

        P_t = P_{t-1} * exp(r_t)
    """
    prices = [anchor]
    for r in log_return_forecasts:
        prices.append(prices[-1] * np.exp(r))
    return prices[1:]


def _sigma_bands(anchor, mu, sigma, steps):
    """
    Compute ±1σ and ±2σ GBM confidence bands around the expected path.

    For a GBM random walk, after t steps from anchor P_0:

        log(P_t / P_0) ~ Normal(mu * t,  sigma² * t)

    So the k-sigma band at step t is:

        P_0 * exp(mu * t  ±  k * sigma * sqrt(t))

    This is the *distributional* spread of where a single random path
    could plausibly end up after t steps, given the estimated parameters.
    t is measured from 1 (the first forecast day), so the band width
    is zero at the anchor and grows as sqrt(t) into the future.

    Parameters
    ----------
    anchor : float   Last known price (band origin).
    mu     : float   Daily drift.
    sigma  : float   Daily volatility.
    steps  : int     Number of forecast days.

    Returns
    -------
    upper_1, lower_1 : np.ndarray   ±1σ price bands.
    upper_2, lower_2 : np.ndarray   ±2σ price bands.
    """
    t        = np.arange(1, steps + 1, dtype=float)
    center   = mu * t                       # log-scale expected drift
    spread_1 = 1.0 * sigma * np.sqrt(t)    # ±1σ
    spread_2 = 2.0 * sigma * np.sqrt(t)    # ±2σ

    upper_1 = anchor * np.exp(center + spread_1)
    lower_1 = anchor * np.exp(center - spread_1)
    upper_2 = anchor * np.exp(center + spread_2)
    lower_2 = anchor * np.exp(center - spread_2)

    return upper_1, lower_1, upper_2, lower_2


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE  (monthly, re-anchored)
# ─────────────────────────────────────────────────────────────────────────────

def _run(df, start, end, mu, sigma):
    """
    Monthly expanding-window random walk engine.

    Each iteration covers one calendar month, defined as:
        Jan 1  (anchor)  →  Feb 1  (last forecasted day, inclusive)

    Steps per iteration:
      1. Anchor at the real close on the 1st of the current month
         (observable at forecast time; last available close handles weekends).
      2. Re-estimate mu on the expanding training set.
      3. Forecast the expected path (r_t = mu) for Jan 1 → Feb 1 inclusive.
      4. Compute ±1σ / ±2σ bands from the same anchor.
      5. Append results (forecast + real + bands for every day including Feb 1).
      6. Grow training set.  Next iteration anchors at Feb 1's real close,
         which was just appended — no separate re-anchoring step needed.

    Returns
    -------
    pd.DataFrame  Columns: Date | Forecast | Real | Upper1 | Lower1 | Upper2 | Lower2
    """
    all_forecasts = []
    current_date  = start
    current_train = df[df["Date"] < start].copy()

    while current_date <= end:

        # ── Month boundaries ─────────────────────────────────────────────────
        # next_month is the 1st of the following month, which is the LAST day
        # included in the forecast slice (forecast runs from current_date up
        # to and INCLUDING next_month, i.e. Jan 1 → Feb 1).
        next_month = current_date + pd.offsets.MonthBegin(1)

        # ── Test slice: current 1st through next 1st, inclusive ──────────────
        test_slice = df[
            (df["Date"] >= current_date) &
            (df["Date"] <= next_month)
        ].copy()

        if test_slice.empty or len(current_train) < 2:
            current_date = next_month
            continue

        # ── Anchor: real close on the 1st of the current month ───────────────
        # Observable at forecast time; handles weekends/holidays by taking the
        # last available close on or before the 1st.
        anchor_rows  = df[df["Date"] <= current_date]
        anchor_price = (
            anchor_rows["Close"].iloc[-1]
            if not anchor_rows.empty
            else current_train["Close"].iloc[-1]
        )

        # ── Re-estimate mu on expanding training set ─────────────────────────
        mu_now, sigma_now = _estimate_params(current_train["Log_Return"])

        # ── Expected path ────────────────────────────────────────────────────
        steps            = len(test_slice)
        forecast_returns = np.full(steps, mu_now)
        forecast_prices  = _reconstruct_prices(anchor_price, forecast_returns)

        # ── Sigma bands from same anchor ─────────────────────────────────────
        upper_1, lower_1, upper_2, lower_2 = _sigma_bands(
            anchor_price, mu_now, sigma_now, steps
        )

        all_forecasts.append(pd.DataFrame({
            "Date":     test_slice["Date"].values,
            "Forecast": forecast_prices,
            "Real":     test_slice["Close"].values,
            "Upper1":   upper_1,
            "Lower1":   lower_1,
            "Upper2":   upper_2,
            "Lower2":   lower_2,
        }))

        # ── Grow training set, then move to next month ────────────────────────
        # Only append days strictly after current_date — the anchor day (1st
        # of current month) is already in current_train, and next_month (Feb 1)
        # will become the next anchor and must not be duplicated.
        new_rows = test_slice[
            (test_slice["Date"] > current_date) &
            (test_slice["Date"] < next_month)
        ]
        current_train = pd.concat([current_train, new_rows], ignore_index=True)
        current_date  = next_month

    return pd.concat(all_forecasts, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def forecast_rw_monthly(name, df, year_n):
    """
    Monthly expanding-window random walk for a single asset.

    The price path is re-anchored at the start of each calendar month and
    mu is re-estimated on the growing training set.  Sigma bands reset with
    each new anchor (t counts from 0 at each month boundary).

    This mirrors forecast_final() from final_arima.py.

    Returns
    -------
    train       : DataFrame
    forecast_df : DataFrame  Date|Forecast|Real|Upper1|Lower1|Upper2|Lower2
    params      : tuple      (mu, sigma) from the full pre-year_n training set.
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n}-12-01")   # last iteration: Dec 1 → Jan 1
    train = df[df["Date"] < start].copy()

    mu, sigma = _estimate_params(train["Log_Return"])

    print(f"\n{'=' * 60}")
    print(f"  {name}  —  Random Walk  (monthly re-anchored)")
    print(f"{'=' * 60}")
    print(f"  mu    (daily drift) : {mu:.6f}")
    print(f"  sigma (daily vol.)  : {sigma:.6f}")

    forecast_df = _run(df, start, end, mu, sigma)

    return train, forecast_df, (round(mu, 6), round(sigma, 6))


def forecast_rw_static(name, df, year_n):
    """
    Static (one-shot) random walk forecast for a single asset.

    mu and sigma are estimated once on all data before year_n.  The expected
    path propagates freely from the last known close for the entire year.
    Sigma bands grow from that single anchor for the full 365 days.

    This mirrors forecast_static() from final_arima.py.

    Returns
    -------
    train       : DataFrame
    forecast_df : DataFrame  Date|Forecast|Real|Upper1|Lower1|Upper2|Lower2
    params      : tuple      (mu, sigma)
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    mu, sigma = _estimate_params(train["Log_Return"])

    print(f"\n{'=' * 60}")
    print(f"  {name}  —  Random Walk  (static / one-shot)")
    print(f"{'=' * 60}")
    print(f"  mu    (daily drift) : {mu:.6f}")
    print(f"  sigma (daily vol.)  : {sigma:.6f}")

    anchor_price     = train["Close"].iloc[-1]
    steps            = len(test)
    forecast_returns = np.full(steps, mu)
    forecast_prices  = _reconstruct_prices(anchor_price, forecast_returns)

    upper_1, lower_1, upper_2, lower_2 = _sigma_bands(anchor_price, mu, sigma, steps)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
        "Upper1":   upper_1,
        "Lower1":   lower_1,
        "Upper2":   upper_2,
        "Lower2":   lower_2,
    })

    return train, forecast_df, (round(mu, 6), round(sigma, 6))


def forecast_rw_longrun(name, df, start_year, end_year):
    """
    Long-run static random walk across multiple years in a single shot.

    mu and sigma are estimated on all data before start_year.  The expected
    path and sigma bands propagate freely from the last known close all the
    way through end_year — no retraining, no re-anchoring.

    This mirrors forecast_static_longrun() from final_arima.py.

    Returns
    -------
    train       : DataFrame
    forecast_df : DataFrame  Date|Forecast|Real|Upper1|Lower1|Upper2|Lower2
    params      : tuple      (mu, sigma)
    """
    start = pd.Timestamp(f"{start_year}-01-01")
    end   = pd.Timestamp(f"{end_year + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    mu, sigma = _estimate_params(train["Log_Return"])

    print(f"\n{'=' * 60}")
    print(f"  {name}  —  Random Walk  (long-run · {start_year}→{end_year})")
    print(f"{'=' * 60}")
    print(f"  mu    (daily drift) : {mu:.6f}")
    print(f"  sigma (daily vol.)  : {sigma:.6f}")

    anchor_price     = train["Close"].iloc[-1]
    steps            = len(test)
    forecast_returns = np.full(steps, mu)
    forecast_prices  = _reconstruct_prices(anchor_price, forecast_returns)

    upper_1, lower_1, upper_2, lower_2 = _sigma_bands(anchor_price, mu, sigma, steps)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
        "Upper1":   upper_1,
        "Lower1":   lower_1,
        "Upper2":   upper_2,
        "Lower2":   lower_2,
    })

    # ── Final day comparison ──────────────────────────────────────────────────
    last  = forecast_df.iloc[-1]
    error = ((last["Forecast"] - last["Real"]) / last["Real"]) * 100
    print(f"\n  Final day  :  {last['Date'].date()}")
    if name == "EUR/USD":
        print(f"  Forecast   :  {last['Forecast']:.4f}")
        print(f"  Real       :  {last['Real']:.4f}")
    else:
        print(f"  Forecast   :  {last['Forecast']:.2f}")
        print(f"  Real       :  {last['Real']:.2f}")
    print(f"  Error      :  {error:.2f}%")

    return train, forecast_df, (round(mu, 6), round(sigma, 6))