"""
future.py
=========

True out-of-sample forecast of every asset from the end of the available
data (1 June 2026, inclusive) up to 1 January 2027.

Why this file exists separately
-------------------------------
Every model elsewhere in the project is evaluated against KNOWN future prices,
which lets the monthly variants re-anchor to the real close at the start of
each month and retrain on the data that has since arrived.  Here there is no
such data: 1 June 2026 is the last real observation.  So every model must run
in an ADAPTED STATIC mode:

    * fit ONCE on all data up to the anchor (last real close),
    * propagate forward in a single uninterrupted pass to 1 Jan 2027,
    * never re-anchor, never retrain.

This is exactly the discipline of the *static* / *long-run* models, reused
here with the forecast horizon set to the future instead of to a historical
test year.  ARIMA and the random walk propagate a fixed drift; XGBoost and the
LSTM propagate recursively (each prediction feeds the next).

Only three checkpoints are printed per asset: 1 July 2026, 1 October 2026,
and 1 January 2027.

Public API
----------
forecast_future(name, df)  ->  (history_df, forecast_df, info)
    history_df  : real data up to the anchor (Date, Close)
    forecast_df : Date | <one column per model>   (forecast prices only)
    info        : dict of per-model parameters/orders chosen on the fit
"""

import pandas as pd
import numpy as np

# Reuse the EXACT model internals already validated in the project, so the
# future forecast is produced by the same code paths as the rest of the thesis.
from final_arima import select_order, _make_series, _fit as _fit_arima, _reconstruct_prices
from randomwalk  import _estimate_params, _reconstruct_prices as _rw_reconstruct, _sigma_bands
from boosting    import (_training_frame   as _xgb_training_frame,
                         _select_params    as _xgb_select_params,
                         _fit              as _xgb_fit,
                         _recursive_path   as _xgb_recursive_path)
from rnn         import (_training_frame   as _rnn_training_frame,
                         _select_config    as _rnn_select_config,
                         _recursive_path   as _rnn_recursive_path,
                         DEVICE, _seed_everything)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

HORIZON_END   = pd.Timestamp("2027-01-01")   # last day to forecast (inclusive)
CHECKPOINTS   = [pd.Timestamp("2026-07-01"),
                 pd.Timestamp("2026-10-01"),
                 pd.Timestamp("2027-01-01")]


def _future_dates(anchor_date, end=HORIZON_END):
    """Continuous daily index from the day AFTER the anchor up to `end`."""
    start = anchor_date + pd.Timedelta(days=1)
    return pd.date_range(start=start, end=end, freq="D")


# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL FUTURE FORECASTERS  (all adapted-static: one fit, single pass)
# ─────────────────────────────────────────────────────────────────────────────

def _future_arima(df, anchor_price, future_dates, criterion="aic"):
    """Fixed-order ARIMA on log returns; fixed-drift propagation."""
    best_order, _ = select_order(df["Log_Return"], df["Date"], criterion=criterion)
    series        = _make_series(df["Log_Return"], df["Date"])
    fitted        = _fit_arima(series, best_order)

    returns = fitted.forecast(steps=len(future_dates))
    prices  = _reconstruct_prices(anchor_price, returns)
    return prices, {"order": best_order}


def _future_rw(df, anchor_price, future_dates):
    """Random walk: expected path along the estimated drift, plus sigma bands."""
    mu, sigma = _estimate_params(df["Log_Return"])

    returns = np.full(len(future_dates), mu)
    prices  = _rw_reconstruct(anchor_price, returns)

    u1, l1, u2, l2 = _sigma_bands(anchor_price, mu, sigma, len(future_dates))
    bands = {"Upper1": u1, "Lower1": l1, "Upper2": u2, "Lower2": l2}
    return prices, {"mu": round(mu, 6), "sigma": round(sigma, 6)}, bands


def _future_xgb(df, anchor_price, future_dates):
    """XGBoost: one fit on all data, recursive propagation."""
    feats        = _xgb_training_frame(df)
    best_params, _ = _xgb_select_params(feats)
    model        = _xgb_fit(feats, best_params)

    history = df[["Date", "Close", "Log_Return"]].tail(80)
    prices  = _xgb_recursive_path(model, history, future_dates, anchor_price)
    return prices, {"params": best_params}


def _future_rnn(df, anchor_price, future_dates):
    """LSTM: one config selection + fit, recursive propagation."""
    _seed_everything()
    feats = _rnn_training_frame(df)
    best_config, _, best_bundle = _rnn_select_config(feats, DEVICE)

    history = df[["Date", "Close", "Log_Return"]].tail(best_config["seq_len"] + 60)
    prices  = _rnn_recursive_path(best_bundle, history, future_dates, anchor_price)
    return prices, {"config": best_config}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def forecast_future(name, df):
    """
    Forecast `df` from its last real close to 1 Jan 2027 with all four models.

    Returns
    -------
    history_df  : DataFrame  (Date, Close)  real data up to the anchor.
    forecast_df : DataFrame  Date | ARIMA | RW | XGBoost | LSTM  (prices).
    info        : dict        per-model fitted parameters + the RW sigma bands.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # Anchor = last real observation (1 June 2026).
    anchor_row   = df.iloc[-1]
    anchor_date  = pd.Timestamp(anchor_row["Date"])
    anchor_price = float(anchor_row["Close"])

    future_dates = _future_dates(anchor_date)

    print(f"\n{'=' * 60}")
    print(f"  {name}  —  FUTURE forecast (adapted static)")
    print(f"  Anchor : {anchor_date.date()}  @  {anchor_price:,.4f}")
    print(f"  Horizon: {future_dates[0].date()} → {future_dates[-1].date()}")
    print(f"{'=' * 60}")

    arima_p, arima_info       = _future_arima(df, anchor_price, future_dates)
    rw_p,    rw_info, rw_bands = _future_rw(df, anchor_price, future_dates)
    xgb_p,   xgb_info         = _future_xgb(df, anchor_price, future_dates)
    rnn_p,   rnn_info         = _future_rnn(df, anchor_price, future_dates)

    forecast_df = pd.DataFrame({
        "Date":    future_dates,
        "ARIMA":   arima_p,
        "RW":      rw_p,
        "XGBoost": xgb_p,
        "LSTM":    rnn_p,
    })
    for k, v in rw_bands.items():          # attach RW confidence bands
        forecast_df[k] = v

    history_df = df[["Date", "Close"]].copy()
    info = {"ARIMA": arima_info, "RW": rw_info,
            "XGBoost": xgb_info, "LSTM": rnn_info}

    _print_checkpoints(name, forecast_df)
    return history_df, forecast_df, info


def _print_checkpoints(name, forecast_df):
    """Print ONLY the three requested checkpoints: Jul, Oct, next Jan."""
    is_fx  = (name == "EUR/USD")
    models = ["ARIMA", "RW", "XGBoost", "LSTM"]

    print(f"\n  {name} — forecast at checkpoints")
    print("  " + "-" * 56)
    header = f"  {'Date':<12}" + "".join(f"{m:>11}" for m in models)
    print(header)

    for d in CHECKPOINTS:
        row = forecast_df[forecast_df["Date"] == d]
        if row.empty:
            print(f"  {d.date()}  (no forecast)")
            continue
        cells = ""
        for m in models:
            v = row[m].values[0]
            cells += f"{v:>11.4f}" if is_fx else f"{v:>11.2f}"
        print(f"  {str(d.date()):<12}{cells}")