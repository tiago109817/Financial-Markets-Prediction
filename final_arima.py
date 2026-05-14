import pandas as pd
import numpy as np
import itertools
import warnings

from statsmodels.tsa.arima.model import ARIMA


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_series(log_returns, dates):
    """
    Build a daily-frequency pd.Series from log returns + dates.

    data.py already guarantees a continuous daily index with no gaps
    (weekends and holidays are forward-filled there), so all we need
    here is to drop the leading NaN from the first log return and set
    the frequency so ARIMA sees a regular series.
    """
    return (
        pd.Series(log_returns.values, index=pd.DatetimeIndex(dates))
        .dropna()
        .asfreq("D")
    )


def _fit(series, order):
    """
    Fit ARIMA(p,d,q) on a pre-built daily Series.
    Convergence warnings are suppressed — failures raise normally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(
            series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(method_kwargs={"maxiter": 300})


def _reconstruct_prices(anchor, log_return_forecasts):
    """
    Convert a sequence of forecasted log returns into a price path,
    starting from `anchor`.

        P_t = P_{t-1} * exp(r_t)
    """
    prices = [anchor]
    for r in log_return_forecasts:
        prices.append(prices[-1] * np.exp(r))
    return prices[1:]


# ─────────────────────────────────────────────────────────────────────────────
# ORDER SELECTION  (AIC / BIC grid search)
# ─────────────────────────────────────────────────────────────────────────────

def select_order(log_returns, dates,
                 p_range=range(0, 4),
                 d_range=range(0, 2),
                 q_range=range(0, 4),
                 criterion="aic"):
    """
    Grid-search over (p, d, q) and return the order that minimises
    the chosen information criterion on the supplied training series.

    ── ARIMA order recap ────────────────────────────────────────────────────
    An ARIMA(p, d, q) model has three structural parameters:

      p  (AR order) — how many lagged values of the series itself are used
                      as predictors.  A high p means the model relies on a
                      long memory of past returns.

      d  (integration order) — how many times the series must be differenced
                      to become stationary.  Log returns are already close to
                      stationary, so d = 0 or 1 is almost always sufficient.

      q  (MA order) — how many lagged forecast errors are included.  This
                      captures short-lived shocks (e.g. a surprise macro
                      announcement) that decay quickly.

    ── Why AIC / BIC? ───────────────────────────────────────────────────────
    Adding more parameters (higher p or q) always improves in-sample fit,
    but risks overfitting — the model memorises noise rather than learning
    signal.  Information criteria penalise complexity explicitly:

      AIC  =  -2 · log-likelihood  +  2k
      BIC  =  -2 · log-likelihood  +  k · log(n)

    where k is the number of free parameters and n the number of observations.
    The model with the LOWEST score wins.  BIC applies a heavier penalty than
    AIC (log(n) > 2 for n > 7), so it tends to select sparser models.
    AIC is generally preferred when the goal is prediction accuracy; BIC when
    interpretability or parsimony matters more.

    ── Search strategy ──────────────────────────────────────────────────────
    We do an exhaustive grid search: every combination of (p, d, q) within
    the supplied ranges is fitted and scored.  Models that fail to converge
    (singular covariance matrix, non-invertible roots, etc.) are silently
    skipped — they would not be reliable in production anyway.

    Parameters
    ----------
    log_returns : pd.Series   Log-return series (training data).
    dates       : pd.Series   Corresponding dates.
    p_range     : iterable    AR orders to try   (default 0–3).
    d_range     : iterable    Integration orders (default 0–1).
    q_range     : iterable    MA orders to try   (default 0–3).
    criterion   : str         "aic" or "bic".

    Returns
    -------
    best_order  : tuple (p, d, q)
    grid        : pd.DataFrame  Full ranked grid (all converged models).
    """
    series     = _make_series(log_returns, dates)
    best_order = None
    best_score = np.inf
    records    = []

    for p, d, q in itertools.product(p_range, d_range, q_range):

        # ARIMA(0,0,0) is a white-noise model with no predictive structure —
        # there is nothing to estimate, so we skip it.
        if p == 0 and d == 0 and q == 0:
            continue

        try:
            m = _fit(series, (p, d, q))

            # Extract the criterion score for this combination.
            # Lower is better for both AIC and BIC.
            score = m.aic if criterion == "aic" else m.bic

            records.append({"p": p, "d": d, "q": q,
                             "aic": m.aic, "bic": m.bic})

            if score < best_score:
                best_score = score
                best_order = (p, d, q)

        except Exception:
            # Some (p,d,q) combinations produce singular parameter matrices
            # or fail to converge within the iteration limit — skip silently.
            pass

    # Return the full grid sorted by the chosen criterion so the caller can
    # inspect all candidates, not just the winner.
    grid = (
        pd.DataFrame(records)
        .sort_values(criterion)
        .reset_index(drop=True)
    )
    return best_order, grid


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _run(df, start, end, order):
    """
    Monthly expanding-window ARIMA engine.

    For each calendar month in [start, end]:
      1. Fit ARIMA(order) on all training data seen so far.
      2. Anchor the price forecast at the last available close on or
         before the 1st of the current month (most-recent trading day
         if the 1st falls on a weekend / holiday).
      3. Forecast log returns for every day in the month and convert
         them back to prices via the anchor.
      4. Append the month's results and grow the training set.

    Parameters
    ----------
    df    : DataFrame  Columns: Date, Close, Log_Return.
    start : Timestamp  First day of the forecast horizon.
    end   : Timestamp  Last  day of the forecast horizon.
    order : tuple      (p, d, q) — fixed for the entire run.

    Returns
    -------
    pd.DataFrame  Columns: Date | Forecast | Real
    """
    all_forecasts = []
    current_date  = start
    current_train = df[df["Date"] < start].copy()

    while current_date <= end:

        # ── Month boundaries ─────────────────────────────────────────────────
        next_month = current_date + pd.offsets.MonthBegin(1)
        if next_month > end:
            next_month = end + pd.Timedelta(days=1)

        # ── Test slice ───────────────────────────────────────────────────────
        test_slice = df[
            (df["Date"] >= current_date) &
            (df["Date"] <  next_month)
        ].copy()

        if test_slice.empty or len(current_train) < 50:
            current_date = next_month
            continue

        # ── 1st-of-month anchor ──────────────────────────────────────────────
        # Take the last available close on or before the 1st of the month.
        # This handles weekends and public holidays automatically.
        first_of_month = current_date.replace(day=1)
        anchor_rows    = df[df["Date"] <= first_of_month]
        anchor_price   = (
            anchor_rows["Close"].iloc[-1]
            if not anchor_rows.empty
            else current_train["Close"].iloc[-1]   # safety fallback
        )

        # ── Fit ──────────────────────────────────────────────────────────────
        series = _make_series(current_train["Log_Return"], current_train["Date"])
        fitted = _fit(series, order)

        # ── Forecast ─────────────────────────────────────────────────────────
        forecast_returns = fitted.forecast(steps=len(test_slice))
        forecast_prices  = _reconstruct_prices(anchor_price, forecast_returns)

        all_forecasts.append(pd.DataFrame({
            "Date":     test_slice["Date"].values,
            "Forecast": forecast_prices,
            "Real":     test_slice["Close"].values,
        }))

        # ── Grow training set ────────────────────────────────────────────────
        current_train = pd.concat([current_train, test_slice], ignore_index=True)
        current_date  = next_month

    return pd.concat(all_forecasts, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def forecast_final(name, df, year_n, criterion="aic"):
    """
    Full monthly ARIMA pipeline for a single asset.

    Steps
    -----
    1. Split data into training (everything before year_n) and forecast horizon.
    2. Run AIC/BIC grid search on the training set to find the best (p,d,q).
    3. Print the selection results (top 5 models + chosen order).
    4. Run the monthly expanding-window engine with the chosen order.

    Parameters
    ----------
    name      : str   Asset label (used in printed output).
    df        : DataFrame
    year_n    : int   Year to forecast (e.g. 2024).
    criterion : str   "aic" (default) or "bic".

    Returns
    -------
    train       : DataFrame   Historical data up to start of year_n.
    forecast_df : DataFrame   Date | Forecast | Real  for the full year.
    best_order  : tuple       (p, d, q) chosen by the criterion.
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n + 1}-01-01")
    train = df[df["Date"] < start].copy()

    # ── Order selection ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  ARIMA order selection  ({criterion.upper()})")
    print(f"{'=' * 60}")

    best_order, grid = select_order(
        train["Log_Return"],
        train["Date"],
        criterion=criterion,
    )

    print(f"  Best order : ARIMA{best_order}")
    print(f"  Top 5 candidates:")
    print(grid.head(5).to_string(index=False))

    # ── Monthly forecast ──────────────────────────────────────────────────────
    forecast_df = _run(df, start, end, best_order)

    return train, forecast_df, best_order


def forecast_static(name, df, year_n, criterion="aic"):
    """
    Static ARIMA forecast for a single asset.

    The model is fitted once on all data before year_n and then used to
    forecast the entire year in a single shot — no retraining, no monthly
    updates.  This is the simplest possible ARIMA strategy and serves as
    a long-term baseline to compare against the monthly model and ML.

    The key structural difference vs forecast_final:
      - Monthly : retrained every month on growing data, anchor reset each month.
      - Static  : fitted once, anchor set at the last training close, forecast
                  propagates freely for 365 days.  Errors can accumulate over
                  time, which is precisely what makes it an honest test of
                  long-term predictive power.

    Steps
    -----
    1. Split data into training (everything before year_n) and test (full year).
    2. Run AIC/BIC grid search on the training set to find the best (p,d,q).
    3. Print the selection results (top 5 models + chosen order).
    4. Fit once and forecast all steps in one call.

    Parameters
    ----------
    name      : str   Asset label (used in printed output).
    df        : DataFrame
    year_n    : int   Year to forecast (e.g. 2024).
    criterion : str   "aic" (default) or "bic".

    Returns
    -------
    train       : DataFrame   Historical data up to start of year_n.
    forecast_df : DataFrame   Date | Forecast | Real  for the full year.
    best_order  : tuple       (p, d, q) chosen by the criterion.
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    # ── Order selection ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  ARIMA order selection  ({criterion.upper()})")
    print(f"{'=' * 60}")

    best_order, grid = select_order(
        train["Log_Return"],
        train["Date"],
        criterion=criterion,
    )

    print(f"  Best order : ARIMA{best_order}")
    print(f"  Top 5 candidates:")
    print(grid.head(5).to_string(index=False))

    # ── Single fit on the entire training set ─────────────────────────────────
    series = _make_series(train["Log_Return"], train["Date"])
    fitted = _fit(series, best_order)

    # ── Print fitted parameters ───────────────────────────────────────────────
    print(f"\n  Fitted parameters  (ARIMA{best_order}):")
    print(f"  AIC : {fitted.aic:.2f}   BIC : {fitted.bic:.2f}")
    print(fitted.params.to_string())

    # ── One-shot forecast for the full year ───────────────────────────────────
    # Anchor at the last known price before the forecast horizon.
    # From here the price path propagates freely — no re-anchoring.
    anchor_price     = train["Close"].iloc[-1]
    forecast_returns = fitted.forecast(steps=len(test))
    forecast_prices  = _reconstruct_prices(anchor_price, forecast_returns)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
    })

    return train, forecast_df, best_order

def forecast_static_longrun(name, df, start_year, end_year, criterion="aic"):
    """
    Long-run static ARIMA forecast across multiple years in a single shot.

    The model is fitted once on all data before start_year, then forecasts
    every single day from start_year all the way through to end_year — with
    no retraining and no re-anchoring at any point.  The price path propagates
    freely from the last known close before start_year.

    This is intentionally the most demanding test of ARIMA's long-term
    predictive power.  Any structural drift, trend change, or regime shift
    in the real data will be visible as a growing divergence between the
    forecast and reality — which is precisely what makes it a compelling
    visual for the thesis.

    Parameters
    ----------
    name       : str   Asset label (used in printed output).
    df         : DataFrame
    start_year : int   First year of the forecast horizon (e.g. 2020).
    end_year   : int   Last  year of the forecast horizon (e.g. 2026).
    criterion  : str   "aic" (default) or "bic".

    Returns
    -------
    train       : DataFrame   Historical data up to start of start_year.
    forecast_df : DataFrame   Date | Forecast | Real  for the full horizon.
    best_order  : tuple       (p, d, q) chosen by the criterion.
    """
    start = pd.Timestamp(f"{start_year}-01-01")
    end   = pd.Timestamp(f"{end_year + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    # ── Order selection ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  ARIMA order selection  ({criterion.upper()})")
    print(f"  Forecast horizon: {start_year} → {end_year}")
    print(f"{'=' * 60}")

    best_order, grid = select_order(
        train["Log_Return"],
        train["Date"],
        criterion=criterion,
    )

    print(f"  Best order : ARIMA{best_order}")
    print(f"  Top 5 candidates:")
    print(grid.head(5).to_string(index=False))

    # ── Single fit on the entire training set ─────────────────────────────────
    series = _make_series(train["Log_Return"], train["Date"])
    fitted = _fit(series, best_order)

    # ── Print fitted parameters ───────────────────────────────────────────────
    print(f"\n  Fitted parameters  (ARIMA{best_order}):")
    print(f"  AIC : {fitted.aic:.2f}   BIC : {fitted.bic:.2f}")
    print(fitted.params.to_string())

    # ── One-shot forecast across the full multi-year horizon ──────────────────
    # Anchor at the last known price before start_year. The path then
    # propagates freely for the entire horizon — no monthly resets.
    anchor_price     = train["Close"].iloc[-1]
    forecast_returns = fitted.forecast(steps=len(test))
    forecast_prices  = _reconstruct_prices(anchor_price, forecast_returns)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
    })


    # ── Final day comparison ──────────────────────────────────────────────────
    last  = forecast_df.iloc[-1]
    error = ((last['Forecast'] - last['Real']) / last['Real']) * 100
    print(f"\n  Final day  :  {last['Date'].date()}")
    if name == 'EUR/USD':
        print(f"  Forecast   :  {last['Forecast']:.4f}")
        print(f"  Real       :  {last['Real']:.4f}")
    else:
        print(f"  Forecast   :  {last['Forecast']:.2f}")
        print(f"  Real       :  {last['Real']:.2f}")
    print(f"  Error      :  {error:.2f}%")
    return train, forecast_df, best_order