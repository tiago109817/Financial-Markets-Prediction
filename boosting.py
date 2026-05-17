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
    Build the feature matrix from the available data.

    Features used
    -------------
    From data.py (already computed):
      - Log_Return           : log return at time t

    Derived from Log_Return and Close (computed here):
      - lag_1  … lag_21      : lagged log returns (t-1, t-2, t-5, t-10, t-21)
                               capture yesterday, short memory, weekly, bi-weekly,
                               and monthly autocorrelation structure.
      - vol_5  / vol_21      : rolling std of log returns over 5 and 21 days —
                               a proxy for short- and medium-term volatility.
                               This is the key derived feature analogous to how
                               log returns were derived from Close in data.py.
      - is_filled            : 1 if the date was a weekend / holiday that was
                               forward-filled in data.py, 0 otherwise.
                               Constructed from the date alone (no extra data).

    Target
    ------
      - next_return          : Log_Return shifted by -1  (what we want to predict).
                               Reconstructed to prices via _reconstruct_prices.

    All rows with any NaN (from lagging / rolling) are dropped before returning.
    """
    f = df.copy()

    # ── Lag features (log returns) ────────────────────────────────────────────
    for lag in [1, 2, 5, 10, 21]:
        f[f"lag_{lag}"] = f["Log_Return"].shift(lag)

    # ── Rolling volatility ────────────────────────────────────────────────────
    f["vol_5"]  = f["Log_Return"].rolling(5).std()
    f["vol_21"] = f["Log_Return"].rolling(21).std()

    # ── Is-filled flag ────────────────────────────────────────────────────────
    # Weekends (Saturday = 5, Sunday = 6) were forward-filled in data.py.
    # Public holidays are harder to detect without a calendar library, but
    # weekends account for the vast majority of filled days.
    f["is_filled"] = (pd.to_datetime(f["Date"]).dt.dayofweek >= 5).astype(int)

    # ── Target: next day's log return ─────────────────────────────────────────
    f["next_return"] = f["Log_Return"].shift(-1)

    f = f.dropna().reset_index(drop=True)
    return f


FEATURE_COLS = ["lag_1", "lag_2", "lag_5", "lag_10", "lag_21",
                "vol_5", "vol_21", "is_filled"]
TARGET_COL   = "next_return"


# ─────────────────────────────────────────────────────────────────────────────
# PRICE RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_prices(anchor, log_return_forecasts):
    """
    Convert a sequence of forecasted log returns into a price path.

        P_t = P_{t-1} * exp(r_t)

    Identical to final_arima.py so results are directly comparable.
    """
    prices = [anchor]
    for r in log_return_forecasts:
        prices.append(prices[-1] * np.exp(r))
    return prices[1:]


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_params(features_df, n_splits=5):
    """
    Select XGBoost hyperparameters via time-series cross-validation.

    Why time-series CV?
    -------------------
    Standard k-fold shuffles the data, so future observations can end up
    in the training fold — a form of look-ahead bias that inflates scores.
    TimeSeriesSplit always trains on the past and validates on the future,
    preserving the causal structure of the problem.

    Grid searched
    -------------
      n_estimators  : number of boosting rounds (trees).
      max_depth     : maximum depth of each tree.
      learning_rate : shrinkage applied to each tree's contribution.
                      Lower rate + more trees usually generalises better,
                      but takes longer.

    The grid is intentionally compact — exhaustive search on a large grid
    would be expensive and is unnecessary for a BSc thesis baseline.

    Scoring metric: mean absolute error (MAE) on log returns.
    MAE is preferred over MSE here because log returns have occasional
    large outliers (crashes, rallies) that would dominate squared error.

    Parameters
    ----------
    features_df : DataFrame   Output of _build_features on the training set.
    n_splits    : int         Number of CV folds (default 5).

    Returns
    -------
    best_params : dict   Best hyperparameter combination found.
    results_df  : DataFrame   Full grid with mean MAE per combination.
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
                        model = XGBRegressor(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            objective="reg:squarederror",
                            random_state=42,
                            verbosity=0,
                        )
                        model.fit(X_tr, y_tr)

                    preds    = model.predict(X_val)
                    fold_mae = np.mean(np.abs(preds - y_val))
                    fold_maes.append(fold_mae)

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
    """
    Fit an XGBRegressor on the supplied feature DataFrame.

    Parameters
    ----------
    features_df : DataFrame   Output of _build_features.
    params      : dict        Hyperparameters (from select_params).

    Returns
    -------
    Fitted XGBRegressor.
    """
    X = features_df[FEATURE_COLS].values
    y = features_df[TARGET_COL].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _recursive_forecast(model, seed_returns, n_steps):
    """
    Produce n_steps forecasts recursively from a fitted model.

    XGBoost is a cross-sectional model — it cannot natively produce a
    multi-step forecast like ARIMA's .forecast(steps=N).  Instead we:
      1. Predict day t+1 using the current feature vector.
      2. Append the prediction to the recent-return buffer.
      3. Recompute lag and volatility features for day t+2.
      4. Repeat until n_steps are produced.

    This is the standard recursive multi-step strategy and is consistent
    across all three public functions (static, monthly, long-run).

    Parameters
    ----------
    model        : fitted XGBRegressor
    seed_returns : list   Recent log returns used to seed the feature vector.
                          Must contain at least 21 values.
    n_steps      : int    Number of future steps to forecast.

    Returns
    -------
    list of float   Predicted log returns (length = n_steps).
    """
    recent = list(seed_returns[-21:])   # keep a rolling buffer of 21 values
    predicted = []

    for _ in range(n_steps):
        lags  = recent[-21:]
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
        recent.append(pred)   # feed prediction back as next lag_1

    return predicted


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def _print_params(name, params, results_df):
    """
    Print hyperparameter selection results — mirrors ARIMA order-selection
    output in final_arima.py so the console output is consistent.
    """
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  XGBoost hyperparameter selection")
    print(f"{'=' * 60}")
    print(f"  Best params : {params}")
    print(f"  Top 5 candidates (by CV MAE):")
    print(results_df.head(5).to_string(index=False))


def _print_feature_importance(name, model):
    """
    Print normalised feature importance scores.

    XGBoost assigns each feature a score based on how much each feature
    reduces the loss across all trees (gain-based importance).  Useful
    for interpreting which features drive the model's predictions.
    """
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    print(f"\n  Feature importances ({name}):")
    for feat, score in importance.items():
        bar = "█" * int(score * 40)
        print(f"    {feat:<12}  {score:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE  —  monthly expanding window
# ─────────────────────────────────────────────────────────────────────────────

def _run(df, full_features, start, end, params):
    """
    Monthly expanding-window XGBoost engine.

    Mirrors _run() in final_arima.py exactly:

      For each calendar month in [start, end]:
        1. Retrain XGBoost on all feature rows whose Date < current month.
        2. Anchor the price forecast at the last available close on or
           before the 1st of the current month.
        3. Predict next-day log returns recursively for the whole month.
        4. Convert log returns → prices via _reconstruct_prices.
        5. Append results and advance to the next month.

    Parameters
    ----------
    df           : DataFrame   Full asset DataFrame (Date, Close, Log_Return …).
    full_features: DataFrame   Output of _build_features(df).
    start        : Timestamp   First day of the forecast horizon.
    end          : Timestamp   Last  day of the forecast horizon.
    params       : dict        XGBoost hyperparameters.

    Returns
    -------
    pd.DataFrame   Columns: Date | Forecast | Real
    """
    all_forecasts = []
    current_date  = start

    while current_date <= end:

        # ── Month boundaries ─────────────────────────────────────────────────
        next_month = current_date + pd.offsets.MonthBegin(1)
        if next_month > end:
            next_month = end + pd.Timedelta(days=1)

        # ── Training and test slices ──────────────────────────────────────────
        train_feat = full_features[full_features["Date"] < current_date].copy()
        test_slice = df[
            (df["Date"] >= current_date) &
            (df["Date"] <  next_month)
        ].copy()

        if test_slice.empty or len(train_feat) < 50:
            current_date = next_month
            continue

        # ── Fit ──────────────────────────────────────────────────────────────
        model = _fit(train_feat, params)

        # ── 1st-of-month anchor (same logic as final_arima.py) ────────────────
        first_of_month = current_date.replace(day=1)
        anchor_rows    = df[df["Date"] <= first_of_month]
        anchor_price   = (
            anchor_rows["Close"].iloc[-1]
            if not anchor_rows.empty
            else train_feat["Close"].iloc[-1]
        )

        # ── Recursive forecast for this month ─────────────────────────────────
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
      1. Split data into training (before year_n) and forecast horizon.
      2. Build features on the full dataset.
      3. Run CV hyperparameter search on training features.
      4. Run the monthly expanding-window engine with the best params.

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
    train = df[df["Date"] < start].copy()

    full_features = _build_features(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    best_params, results_df = select_params(train_feat)
    _print_params(name, best_params, results_df)

    end         = pd.Timestamp(f"{year_n + 1}-01-01")
    forecast_df = _run(df, full_features, start, end, best_params)

    return train, forecast_df, best_params


def forecast_xgb_static(name, df, year_n):
    """
    Static XGBoost forecast for a single asset.

    Mirrors forecast_static() in final_arima.py:
      - Model is fitted ONCE on all data before year_n.
      - Forecasts the entire year recursively in a single pass.
      - No retraining, no monthly updates.
      - Errors accumulate over time — an honest long-horizon test.

    Parameters
    ----------
    name   : str   Asset label.
    df     : DataFrame
    year_n : int   Year to forecast.

    Returns
    -------
    train       : DataFrame   Historical data up to start of year_n.
    forecast_df : DataFrame   Date | Forecast | Real  for the full year.
    best_params : dict        Hyperparameters chosen by CV.
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    full_features = _build_features(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    best_params, results_df = select_params(train_feat)
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
      - Model is fitted ONCE on all data before start_year.
      - Forecasts every day from start_year through end_year recursively.
      - No retraining, no re-anchoring at any point.
      - The price path propagates freely — the most demanding test of
        long-term predictive power and the most visually striking for
        the thesis.

    Parameters
    ----------
    name       : str   Asset label.
    df         : DataFrame
    start_year : int   First year of the forecast horizon (e.g. 2020).
    end_year   : int   Last  year of the forecast horizon (e.g. 2025).

    Returns
    -------
    train       : DataFrame   Historical data up to start of start_year.
    forecast_df : DataFrame   Date | Forecast | Real  for the full horizon.
    best_params : dict        Hyperparameters chosen by CV.
    """
    start = pd.Timestamp(f"{start_year}-01-01")
    end   = pd.Timestamp(f"{end_year + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    full_features = _build_features(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    # ── Hyperparameter selection ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  XGBoost hyperparameter selection")
    print(f"  Forecast horizon: {start_year} → {end_year}")
    print(f"{'=' * 60}")

    best_params, results_df = select_params(train_feat)
    print(f"  Best params : {best_params}")
    print(f"  Top 5 candidates (by CV MAE):")
    print(results_df.head(5).to_string(index=False))

    # ── Single fit ────────────────────────────────────────────────────────────
    model = _fit(train_feat, best_params)
    _print_feature_importance(name, model)

    # ── Recursive forecast across the full multi-year horizon ─────────────────
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