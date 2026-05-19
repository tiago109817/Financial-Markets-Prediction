import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(df):
    """
    Build the feature matrix from available data.

    All features are derived exclusively from:
      - Date       (from the original Excel, used to flag forward-filled days)
      - Close      (from the original Excel)
      - Log_Return (computed in data.py from Close)

    Features
    --------
    From data.py:
      Log_Return           — log return at time t.  This is the base signal.

    Lagged log returns  (capturing autocorrelation structure):
      lag_1                — yesterday's return (strongest short-term predictor)
      lag_2                — two days ago
      lag_5                — one week ago (~ 5 trading days)
      lag_10               — two weeks ago
      lag_21               — one month ago

    Derived from Log_Return (analogous to how Log_Return was derived from Close):
      vol_5                — rolling 5-day std of log returns (short-term vol)
      vol_21               — rolling 21-day std of log returns (monthly vol)
        Both are computed over a trailing window so there is no look-ahead.
        Volatility is the single most important derived feature for financial
        time series — markets tend to cluster high-vol days together.

    Derived from Date:
      is_filled            — 1 if the row is a weekend / holiday that was
                             forward-filled in data.py, 0 otherwise.
                             Constructed from the date alone (no external data).
                             Forward-filled days have zero *true* return, so
                             their lagged features carry information about
                             the previous trading session rather than new signal.

    Target
    ------
      next_return          — Log_Return shifted by -1 (the value we predict).
                             Converted back to prices via _reconstruct_prices.

    All rows with any NaN (from lagging / rolling) are dropped before returning.
    """
    f = df.copy()

    for lag in [1, 2, 5, 10, 21]:
        f[f"lag_{lag}"] = f["Log_Return"].shift(lag)

    f["vol_5"]  = f["Log_Return"].rolling(5).std()
    f["vol_21"] = f["Log_Return"].rolling(21).std()

    # Weekend = Saturday (5) or Sunday (6) — the bulk of forward-filled days.
    f["is_filled"] = (pd.to_datetime(f["Date"]).dt.dayofweek >= 5).astype(int)

    f["next_return"] = f["Log_Return"].shift(-1)

    f = f.dropna().reset_index(drop=True)
    return f


FEATURE_COLS = ["lag_1", "lag_2", "lag_5", "lag_10", "lag_21",
                "vol_5", "vol_21", "is_filled"]
TARGET_COL   = "next_return"


# ─────────────────────────────────────────────────────────────────────────────
# PRICE RECONSTRUCTION  (identical to final_arima.py and randomwalk.py)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_prices(anchor, log_return_forecasts):
    """
    Convert a sequence of forecasted log returns into a price path.

        P_t = P_{t-1} * exp(r_t)
    """
    prices = [anchor]
    for r in log_return_forecasts:
        prices.append(prices[-1] * np.exp(r))
    return prices[1:]


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SELECTION  (time-series CV)
# ─────────────────────────────────────────────────────────────────────────────

def _select_params(features_df, n_splits=5):
    """
    Select XGBoost hyperparameters via time-series cross-validation.

    Why time-series CV?
    -------------------
    Standard k-fold shuffles the data, so future observations leak into the
    training fold.  TimeSeriesSplit always trains on the past and validates
    on the future, preserving the causal structure of the problem — exactly
    as we do with ARIMA order selection on the training set only.

    Grid searched
    -------------
      n_estimators  : boosting rounds
      max_depth     : tree depth (controls complexity / overfitting)
      learning_rate : shrinkage per tree

    Scoring: mean absolute error (MAE) on log returns.
    MAE is preferred over MSE because financial returns have occasional large
    outliers (crashes, rallies) that would dominate squared error and bias
    the search towards sparser models than necessary.

    Returns
    -------
    best_params : dict
    results_df  : DataFrame  (full grid, sorted by mean CV MAE)
    """
    X = features_df[FEATURE_COLS].values
    y = features_df[TARGET_COL].values

    param_grid = {
        "n_estimators":  [100, 300, 500],
        "max_depth":     [2, 3, 5],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    tscv    = TimeSeriesSplit(n_splits=n_splits)
    records = []

    for n_est in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            for lr in param_grid["learning_rate"]:

                fold_maes = []

                for train_idx, val_idx in tscv.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = XGBRegressor(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            objective="reg:squarederror",
                            random_state=42,
                            verbosity=0,
                        )
                        m.fit(X_tr, y_tr)

                    fold_maes.append(np.mean(np.abs(m.predict(X_val) - y_val)))

                records.append({
                    "n_estimators":  n_est,
                    "max_depth":     depth,
                    "learning_rate": lr,
                    "mean_mae":      np.mean(fold_maes),
                })

    results_df  = pd.DataFrame(records).sort_values("mean_mae").reset_index(drop=True)
    best_row    = results_df.iloc[0]
    best_params = {
        "n_estimators":  int(best_row["n_estimators"]),
        "max_depth":     int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
    }
    return best_params, results_df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────────

def _fit(features_df, params):
    X = features_df[FEATURE_COLS].values
    y = features_df[TARGET_COL].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        m.fit(X, y)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def _print_params(name, best_params, results_df):
    """
    Print hyperparameter selection results.
    Mirrors the ARIMA order-selection output format for consistency.
    """
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  XGBoost hyperparameter selection")
    print(f"{'=' * 60}")
    print(f"  Best params : n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"learning_rate={best_params['learning_rate']}")
    print(f"  Top 5 candidates (by CV MAE):")
    print(results_df.head(5).to_string(index=False))


def _print_feature_importance(name, model):
    """
    Print normalised gain-based feature importances.

    XGBoost assigns each feature a score based on how much it reduces the
    loss across all trees.  This is the gain-based importance — the most
    informative metric for understanding which features drive predictions.
    """
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    print(f"\n  Feature importances ({name}):")
    for feat, score in importance.items():
        bar = "█" * int(score * 40)
        print(f"    {feat:<12}  {score:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE MULTI-STEP PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def _recursive_forecast(model, seed_returns, n_steps):
    """
    Produce n_steps forecasts recursively from a fitted model.

    XGBoost is a cross-sectional model — it cannot natively produce a
    multi-step forecast the way ARIMA's .forecast(steps=N) does.  Instead:
      1. Build the feature vector from the current return buffer.
      2. Predict the next return.
      3. Append the prediction to the buffer (it becomes lag_1 next step).
      4. Repeat until n_steps are produced.

    This is the standard recursive multi-step strategy and is applied
    consistently across all three public functions (monthly, static, longrun).

    Parameters
    ----------
    model        : fitted XGBRegressor
    seed_returns : list   Recent log returns (must have at least 21 values).
    n_steps      : int    Number of future steps to forecast.

    Returns
    -------
    list of float   Predicted log returns (length = n_steps).
    """
    recent    = list(seed_returns[-21:])
    predicted = []

    for _ in range(n_steps):
        lags  = recent
        vol5  = float(np.std(lags[-5:]))  if len(lags) >= 5  else 0.0
        vol21 = float(np.std(lags[-21:])) if len(lags) >= 21 else 0.0

        x = np.array([[
            lags[-1]  if len(lags) >= 1  else 0.0,   # lag_1
            lags[-2]  if len(lags) >= 2  else 0.0,   # lag_2
            lags[-5]  if len(lags) >= 5  else 0.0,   # lag_5
            lags[-10] if len(lags) >= 10 else 0.0,   # lag_10
            lags[-21] if len(lags) >= 21 else 0.0,   # lag_21
            vol5,
            vol21,
            0.0,   # is_filled — unknown for future dates; 0 is a neutral placeholder
        ]])

        pred = float(model.predict(x)[0])
        predicted.append(pred)
        recent.append(pred)

    return predicted


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE  —  monthly expanding window
# ─────────────────────────────────────────────────────────────────────────────

def _run(df, full_features, start, end, params):
    """
    Monthly expanding-window XGBoost engine.

    Mirrors _run() in final_arima.py exactly:

      For each calendar month in [start, end]:
        1. Retrain XGBoost on all feature rows whose Date < current month.
        2. Anchor the price forecast at the real close on the 1st of the
           current month (last available close handles weekends / holidays).
        3. Predict next-day log returns recursively for the whole month.
        4. Convert log returns → prices via _reconstruct_prices.
        5. Append results and advance to the next month.

    Month boundaries:
        Jan 1  (anchor)  →  Feb 1  (last forecasted day, inclusive)

    This is identical to final_arima.py and randomwalk.py so all three
    models are directly comparable.

    Parameters
    ----------
    df           : DataFrame   Full asset DataFrame (Date, Close, Log_Return …).
    full_features: DataFrame   Output of _build_features(df).
    start        : Timestamp   First day of the forecast horizon.
    end          : Timestamp   Last  day of the forecast horizon (Dec 1).
    params       : dict        XGBoost hyperparameters.

    Returns
    -------
    pd.DataFrame   Columns: Date | Forecast | Real
    """
    all_forecasts = []
    current_date  = start

    while current_date <= end:

        # ── Month boundaries ─────────────────────────────────────────────────
        # next_month is the 1st of the following month — the LAST day included
        # in the test slice (Jan 1 → Feb 1, inclusive), mirroring final_arima.
        next_month = current_date + pd.offsets.MonthBegin(1)

        # ── Test slice: current 1st through next 1st, inclusive ──────────────
        test_slice = df[
            (df["Date"] >= current_date) &
            (df["Date"] <= next_month)
        ].copy()

        # ── Training features: all rows strictly before current month ─────────
        train_feat = full_features[full_features["Date"] < current_date].copy()

        if test_slice.empty or len(train_feat) < 50:
            current_date = next_month
            continue

        # ── Anchor: real close on the 1st of the current month ───────────────
        # Observable at forecast time; last available close handles weekends.
        anchor_rows  = df[df["Date"] <= current_date]
        anchor_price = (
            anchor_rows["Close"].iloc[-1]
            if not anchor_rows.empty
            else train_feat["Close"].iloc[-1]
        )

        # ── Fit on expanding training set ─────────────────────────────────────
        model = _fit(train_feat, params)

        # ── Recursive forecast for the full month ─────────────────────────────
        seed_returns      = list(train_feat["Log_Return"].iloc[-21:])
        predicted_returns = _recursive_forecast(model, seed_returns, len(test_slice))
        forecast_prices   = _reconstruct_prices(anchor_price, predicted_returns)

        all_forecasts.append(pd.DataFrame({
            "Date":     test_slice["Date"].values,
            "Forecast": forecast_prices,
            "Real":     test_slice["Close"].values,
        }))

        current_date = next_month

    return pd.concat(all_forecasts, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def forecast_xgb_monthly(name, df, year_n):
    """
    Full monthly XGBoost pipeline for a single asset.

    Mirrors forecast_final() in final_arima.py:
      1. Split: training = all data before year_n; horizon = year_n.
      2. Build features on the full dataset (no leakage — future rows exist
         but their Date is used only to slice during the engine loop).
      3. Run CV hyperparameter search on training features only.
      4. Run the monthly expanding-window engine with the best params.

    The model is retrained every month on growing data; the anchor resets
    to the real close at the start of each month — identical strategy to
    the monthly ARIMA.

    Parameters
    ----------
    name   : str   Asset label (used in printed output).
    df     : DataFrame
    year_n : int   Year to forecast (e.g. 2024).

    Returns
    -------
    train       : DataFrame   Historical data up to start of year_n.
    forecast_df : DataFrame   Date | Forecast | Real  for the full year.
    best_params : dict        Hyperparameters chosen by CV.
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n}-12-01")   # last iteration: Dec 1 → Jan 1
    train = df[df["Date"] < start].copy()

    full_features = _build_features(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    best_params, results_df = _select_params(train_feat)
    _print_params(name, best_params, results_df)

    forecast_df = _run(df, full_features, start, end, best_params)

    return train, forecast_df, best_params


def forecast_xgb_static(name, df, year_n):
    """
    Static (one-shot) XGBoost forecast for a single asset.

    Mirrors forecast_static() in final_arima.py:
      - Model fitted ONCE on all data before year_n.
      - Forecasts the entire year recursively in a single pass.
      - No retraining, no monthly updates.
      - Price errors accumulate over time — honest long-horizon test.

    Parameters
    ----------
    name   : str
    df     : DataFrame
    year_n : int

    Returns
    -------
    train       : DataFrame
    forecast_df : DataFrame   Date | Forecast | Real
    best_params : dict
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    full_features = _build_features(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    best_params, results_df = _select_params(train_feat)
    _print_params(name, best_params, results_df)

    model = _fit(train_feat, best_params)
    _print_feature_importance(name, model)

    anchor_price      = train["Close"].iloc[-1]
    seed_returns      = list(train_feat["Log_Return"].iloc[-21:])
    predicted_returns = _recursive_forecast(model, seed_returns, len(test))
    forecast_prices   = _reconstruct_prices(anchor_price, predicted_returns)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
    })

    return train, forecast_df, best_params


def forecast_xgb_longrun(name, df, start_year, end_year):
    """
    Long-run static XGBoost forecast across multiple years.

    Mirrors forecast_static_longrun() in final_arima.py:
      - Model fitted ONCE on all data before start_year.
      - Forecasts every day from start_year through end_year recursively.
      - No retraining, no re-anchoring at any point.
      - The price path propagates freely — the most demanding test of
        long-term predictive power, and the most visually striking for
        the thesis.

    Parameters
    ----------
    name       : str
    df         : DataFrame
    start_year : int   First year of the forecast horizon (e.g. 2020).
    end_year   : int   Last  year of the forecast horizon (e.g. 2025).

    Returns
    -------
    train       : DataFrame
    forecast_df : DataFrame   Date | Forecast | Real
    best_params : dict
    """
    start = pd.Timestamp(f"{start_year}-01-01")
    end   = pd.Timestamp(f"{end_year + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    full_features = _build_features(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    print(f"\n{'=' * 60}")
    print(f"  {name}  —  XGBoost hyperparameter selection")
    print(f"  Forecast horizon: {start_year} → {end_year}")
    print(f"{'=' * 60}")

    best_params, results_df = _select_params(train_feat)
    print(f"  Best params : n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"learning_rate={best_params['learning_rate']}")
    print(f"  Top 5 candidates (by CV MAE):")
    print(results_df.head(5).to_string(index=False))

    model = _fit(train_feat, best_params)
    _print_feature_importance(name, model)

    anchor_price      = train["Close"].iloc[-1]
    seed_returns      = list(train_feat["Log_Return"].iloc[-21:])
    predicted_returns = _recursive_forecast(model, seed_returns, len(test))
    forecast_prices   = _reconstruct_prices(anchor_price, predicted_returns)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
    })

    # ── Final day comparison (mirrors final_arima.py output) ─────────────────
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

    return train, forecast_df, best_params