import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (single source of truth — used in train AND inference)
# ─────────────────────────────────────────────────────────────────────────────

LAGS        = [1, 2, 3, 5, 10, 21]
MOM_WINDOWS = [5, 21]
VOL_WINDOWS = [5, 21]
MA_WINDOW   = 21
_EPS        = 1e-8    # small constant to avoid divide-by-zero in vol_ratio


def _add_features(df):
    """
    Compute every model feature from a (Date, Close, Log_Return) frame.

    This is the ONLY place features are defined.  Both the training pipeline
    and the recursive forecaster call it, which guarantees the model always
    sees identically-computed inputs.

    Features
    --------
    lag_{n}        Lagged log returns (n = 1,2,3,5,10,21).  Short-term memory:
                   yesterday, the last few days, one week, two weeks, one month.

    mom_{w}        Momentum = rolling SUM of log returns over w days
                   = log(P_t / P_{t-w}).  Captures trend DIRECTION, which the
                   volatility features (a spread measure) cannot.

    vol_{w}        Rolling std of log returns over w days.  Volatility clusters
                   in markets, so this is a strong predictor of the *size* of
                   the next move.

    vol_ratio      vol_5 / vol_21.  >1 means short-term volatility is elevated
                   relative to the monthly baseline — a simple regime flag.

    ma_gap_21      log(Close / 21-day mean of Close).  How far price sits above
                   or below its own recent average — a mean-reversion signal.

    dow            Day-of-week (0=Mon … 6=Sun) of the feature row.  Captures
                   any weak day-of-week effect.

    is_filled      1 if the row is a weekend (forward-filled), else 0.

    Target
    ------
    next_return    Log_Return shifted by -1 — the value the model predicts.

    NaN rows (from lagging / rolling) are NOT dropped here; callers decide,
    because the recursive forecaster only ever reads the final row.
    """
    f = df.copy()
    f["Date"] = pd.to_datetime(f["Date"])

    for n in LAGS:
        f[f"lag_{n}"] = f["Log_Return"].shift(n)

    for w in MOM_WINDOWS:
        f[f"mom_{w}"] = f["Log_Return"].rolling(w).sum()

    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = f["Log_Return"].rolling(w).std()

    f["vol_ratio"] = f["vol_5"] / (f["vol_21"] + _EPS)

    ma = f["Close"].rolling(MA_WINDOW).mean()
    f["ma_gap_21"] = np.log(f["Close"] / ma)

    f["dow"]       = f["Date"].dt.dayofweek
    f["is_filled"] = (f["Date"].dt.dayofweek >= 5).astype(int)

    f["next_return"] = f["Log_Return"].shift(-1)
    return f


FEATURE_COLS = (
    [f"lag_{n}" for n in LAGS]
    + [f"mom_{w}" for w in MOM_WINDOWS]
    + [f"vol_{w}" for w in VOL_WINDOWS]
    + ["vol_ratio", "ma_gap_21", "dow", "is_filled"]
)
TARGET_COL = "next_return"


def _training_frame(df):
    """Full feature frame with all NaN feature/target rows removed."""
    f = _add_features(df)
    return f.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SELECTION  (regularised time-series CV grid)
# ─────────────────────────────────────────────────────────────────────────────

def _select_params(features_df, n_splits=5):
    """
    Select XGBoost hyperparameters via time-series cross-validation.

    Why time-series CV?
    -------------------
    Standard k-fold shuffles the data, so future observations leak into the
    training fold.  TimeSeriesSplit always trains on the past and validates on
    the future — the same discipline as fitting ARIMA only on pre-year_n data.

    Grid
    ----
    n_estimators      number of boosting rounds (trees)
    max_depth         tree depth — controls complexity / overfitting
    learning_rate     shrinkage per tree (low = more conservative)
    subsample         fraction of ROWS each tree sees (row bagging)
    colsample_bytree  fraction of FEATURES each tree sees (column bagging)

    subsample and colsample_bytree are the two most effective XGBoost
    regularisers after depth: by showing each tree only a random slice of the
    data they stop it memorising the noise that dominates daily returns.

    Scoring: mean absolute error (MAE) on log returns.  MAE is preferred over
    MSE because returns have occasional large outliers (crashes, rallies) that
    would otherwise dominate a squared-error objective.

    Returns
    -------
    best_params : dict        lowest-MAE combination
    results_df  : DataFrame    full grid, sorted by mean CV MAE
    """
    X = features_df[FEATURE_COLS].values
    y = features_df[TARGET_COL].values

    param_grid = {
        "n_estimators":     [100, 200, 300, 400],
        "max_depth":        [2, 3, 4],
        "learning_rate":    [0.008, 0.01, 0.03],
        "subsample":        [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }

    tscv    = TimeSeriesSplit(n_splits=n_splits)
    records = []

    for n_est in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            for lr in param_grid["learning_rate"]:
                for sub in param_grid["subsample"]:
                    for col in param_grid["colsample_bytree"]:

                        fold_maes = []
                        for tr_idx, val_idx in tscv.split(X):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                m = XGBRegressor(
                                    n_estimators=n_est,
                                    max_depth=depth,
                                    learning_rate=lr,
                                    subsample=sub,
                                    colsample_bytree=col,
                                    objective="reg:squarederror",
                                    random_state=42,
                                    n_jobs=-1,
                                    verbosity=0,
                                )
                                m.fit(X[tr_idx], y[tr_idx])

                            fold_maes.append(
                                np.mean(np.abs(m.predict(X[val_idx]) - y[val_idx]))
                            )

                        records.append({
                            "n_estimators":     n_est,
                            "max_depth":        depth,
                            "learning_rate":    lr,
                            "subsample":        sub,
                            "colsample_bytree": col,
                            "mean_mae":         np.mean(fold_maes),
                        })

    results_df = pd.DataFrame(records).sort_values("mean_mae").reset_index(drop=True)
    best_row   = results_df.iloc[0]
    best_params = {
        "n_estimators":     int(best_row["n_estimators"]),
        "max_depth":        int(best_row["max_depth"]),
        "learning_rate":    float(best_row["learning_rate"]),
        "subsample":        float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
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
            n_jobs=-1,
            verbosity=0,
        )
        m.fit(X, y)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def _print_params(name, best_params, results_df):
    """Print hyperparameter selection results (mirrors ARIMA order output)."""
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  XGBoost hyperparameter selection")
    print(f"{'=' * 60}")
    print(f"  Best params : n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"learning_rate={best_params['learning_rate']}, "
          f"subsample={best_params['subsample']}, "
          f"colsample_bytree={best_params['colsample_bytree']}")
    print(f"  Top 5 candidates (by CV MAE):")
    print(results_df.head(5).to_string(index=False))


def _print_feature_importance(name, model):
    """
    Print normalised gain-based feature importances — the XGBoost analogue of
    ARIMA's fitted coefficients: it shows which inputs actually drove the model.
    """
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    print(f"\n  Feature importances ({name}):")
    for feat, score in importance.items():
        bar = "█" * int(score * 40)
        print(f"    {feat:<16}  {score:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE MULTI-STEP PREDICTION  (reuses _add_features — no hand-coded vector)
# ─────────────────────────────────────────────────────────────────────────────

def _recursive_path(model, history, future_dates, anchor_price):
    """
    Forecast a price path recursively, one calendar day at a time.

    XGBoost is cross-sectional — it has no native multi-step forecast like
    ARIMA's .forecast(steps=N).  So for each future day we:
      1. Run _add_features on the current (Date, Close, Log_Return) buffer.
      2. Read the LAST row's feature vector (state up to the previous day).
      3. Predict the next log return, convert to a price from the running price.
      4. Append the synthetic (date, price, return) row to the buffer and repeat.

    Because step 1 calls the SAME function used in training, the model always
    receives consistently-computed inputs (no ddof or calendar desync).

    Parameters
    ----------
    model        : fitted XGBRegressor
    history      : DataFrame   Real (Date, Close, Log_Return) tail ending at the
                               anchor day — must hold at least ~22 rows so the
                               21-day windows are valid.
    future_dates : iterable    Dates to forecast (the test slice's dates).
    anchor_price : float       Real close the path starts from.

    Returns
    -------
    list of float   Forecast prices, one per future date.
    """
    buf = history[["Date", "Close", "Log_Return"]].copy()
    buf["Date"] = pd.to_datetime(buf["Date"])

    prev_price = anchor_price
    prices     = []

    for d in pd.to_datetime(pd.Series(list(future_dates))).tolist():
        feats = _add_features(buf).iloc[-1]
        x = feats[FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = float(model.predict(x)[0])

        new_price = prev_price * np.exp(r)
        prices.append(new_price)

        buf = pd.concat(
            [buf, pd.DataFrame({"Date": [d], "Close": [new_price], "Log_Return": [r]})],
            ignore_index=True,
        )
        if len(buf) > 120:                       # keep the buffer small for speed
            buf = buf.iloc[-120:].reset_index(drop=True)

        prev_price = new_price

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE  —  monthly expanding window
# ─────────────────────────────────────────────────────────────────────────────

def _run(df, full_features, start, end, params):
    """
    Monthly expanding-window XGBoost engine.

    Mirrors _run() in final_arima.py exactly:

      For each calendar month in [start, end]:
        1. Retrain XGBoost on all feature rows whose Date < current month.
        2. Anchor the path at the real close on the 1st of the current month
           (last available close handles weekends / holidays).
        3. Recursively forecast next-day log returns for the whole month and
           convert them to prices from the anchor.
        4. Append results, advance to the next month.

    Month boundaries:  Jan 1 (anchor) → Feb 1 (last forecast day, inclusive).

    Returns
    -------
    pd.DataFrame   Columns: Date | Forecast | Real
    """
    all_forecasts = []
    current_date  = start

    while current_date <= end:

        next_month = current_date + pd.offsets.MonthBegin(1)

        test_slice = df[
            (df["Date"] >= current_date) &
            (df["Date"] <= next_month)
        ].copy()

        train_feat = full_features[full_features["Date"] < current_date].copy()

        if test_slice.empty or len(train_feat) < 50:
            current_date = next_month
            continue

        # Anchor: real close on / before the 1st of the current month.
        anchor_rows  = df[df["Date"] <= current_date]
        anchor_price = (
            anchor_rows["Close"].iloc[-1]
            if not anchor_rows.empty
            else train_feat["Close"].iloc[-1]
        )

        model = _fit(train_feat, params)

        # Seed the recursion with real data through the anchor day.
        history = anchor_rows[["Date", "Close", "Log_Return"]].tail(80)

        forecast_prices = _recursive_path(
            model, history, test_slice["Date"].values, anchor_price
        )

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
    Monthly expanding-window XGBoost for a single asset.  Mirrors
    forecast_final() in final_arima.py: retrained every month on growing data,
    re-anchored to the real close at the start of each month.

    Returns
    -------
    train       : DataFrame   Historical data up to start of year_n.
    forecast_df : DataFrame   Date | Forecast | Real  for the full year.
    best_params : dict        Hyperparameters chosen by CV.
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n}-12-01")   # last iteration: Dec 1 → Jan 1
    train = df[df["Date"] < start].copy()

    full_features = _training_frame(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    best_params, results_df = _select_params(train_feat)
    _print_params(name, best_params, results_df)

    forecast_df = _run(df, full_features, start, end, best_params)

    return train, forecast_df, best_params


def forecast_xgb_static(name, df, year_n):
    """
    Static (one-shot) XGBoost forecast.  Mirrors forecast_static() in
    final_arima.py: fitted ONCE on all data before year_n, then the full year
    is forecast recursively in a single pass (errors accumulate freely).

    Returns
    -------
    train, forecast_df (Date | Forecast | Real), best_params
    """
    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    full_features = _training_frame(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    best_params, results_df = _select_params(train_feat)
    _print_params(name, best_params, results_df)

    model = _fit(train_feat, best_params)
    _print_feature_importance(name, model)

    anchor_price    = train["Close"].iloc[-1]
    history         = train[["Date", "Close", "Log_Return"]].tail(80)
    forecast_prices = _recursive_path(model, history, test["Date"].values, anchor_price)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
    })

    return train, forecast_df, best_params


def forecast_xgb_longrun(name, df, start_year, end_year):
    """
    Long-run static XGBoost forecast across multiple years.  Mirrors
    forecast_static_longrun() in final_arima.py: fitted ONCE on all data before
    start_year, then every day through end_year is forecast recursively with no
    retraining and no re-anchoring — the most demanding test of long-term skill.

    Returns
    -------
    train, forecast_df (Date | Forecast | Real), best_params
    """
    start = pd.Timestamp(f"{start_year}-01-01")
    end   = pd.Timestamp(f"{end_year + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    full_features = _training_frame(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    print(f"\n{'=' * 60}")
    print(f"  {name}  —  XGBoost hyperparameter selection")
    print(f"  Forecast horizon: {start_year} → {end_year}")
    print(f"{'=' * 60}")

    best_params, results_df = _select_params(train_feat)
    print(f"  Best params : n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"learning_rate={best_params['learning_rate']}, "
          f"subsample={best_params['subsample']}, "
          f"colsample_bytree={best_params['colsample_bytree']}")
    print(f"  Top 5 candidates (by CV MAE):")
    print(results_df.head(5).to_string(index=False))

    model = _fit(train_feat, best_params)
    _print_feature_importance(name, model)

    anchor_price    = train["Close"].iloc[-1]
    history         = train[["Date", "Close", "Log_Return"]].tail(80)
    forecast_prices = _recursive_path(model, history, test["Date"].values, anchor_price)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
    })

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