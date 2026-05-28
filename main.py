from data import sp500, gold, eur_usd, bitcoin

from visual import (plot_assets, plot_log_returns)

from runners import (
    run_arima_final,
    run_arima_basic,
    run_randomwalk,
    run_xgb,
)

# ─────────────────────────────────────────────
# IMPORT MODELS
# ─────────────────────────────────────────────

from arima import (
    # Archived models — uncomment here and set their flag to True to reactivate
    forecast_asset_monthly,
    forecast_asset,
    forecast_asset_monthly_propagating,
    forecast_asset_rolling_window,
    forecast_asset_rolling_window_propagating,
)

from final_arima import (
    forecast_final,
    forecast_static,
    forecast_static_longrun,
    select_order,
)

from randomwalk import (
    forecast_rw_monthly,
    forecast_rw_static,
    forecast_rw_longrun,
)

from boosting import (
    forecast_xgb_monthly,
    forecast_xgb_static,
    forecast_xgb_longrun,
)

# ─────────────────────────────────────────────
# CONFIGURATION FLAGS
# ─────────────────────────────────────────────

RUN_VISUALS        = False   # plots of price series, log returns, and forecasts

SELECT_ORDERS      = False   # print best ARIMA order per asset (no forecast)
RUN_STATIC_ARIMA   = False   # one-shot forecast for a single year
RUN_LONGRUN_ARIMA  = False   # one-shot forecast across multiple years (2020→2025)
RUN_FINAL_ARIMA    = False   # monthly expanding-window, AIC/BIC order

RUN_RW_STATIC      = False   # random walk — one-shot for a single year
RUN_RW_MONTHLY     = False   # random walk — monthly re-anchored, expanding window
RUN_RW_LONGRUN     = False   # random walk — one-shot across multiple years (2020→2025)

RUN_XGB_MONTHLY    = True    # XGBoost — monthly expanding window (mirrors ARIMA monthly)
RUN_XGB_STATIC     = False   # XGBoost — one-shot for a single year
RUN_XGB_LONGRUN    = False   # XGBoost — one-shot across multiple years (2020→2025)

RUN_MONTHLY        = False   # basic monthly (arima.py)
RUN_STATIC         = False   # archived
RUN_MONTHLY_PROP   = False   # archived
RUN_ROLLING        = False   # archived
RUN_ROLLING_PROP   = False   # archived

year_n     = 2022   # used by single-year models
start_year = 2020   # used by long-run models
end_year   = 2025   # used by long-run models

# ─────────────────────────────────────────────
# Assets
# ─────────────────────────────────────────────

assets = {
    "Bitcoin": bitcoin,
    "SP500":   sp500,
    "Gold":    gold,
    "EUR/USD": eur_usd,
}

# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────

if RUN_VISUALS:
    plot_assets(assets, mode="close")
    plot_assets(assets, mode="ma")
    # plot_log_returns(assets)

# ─────────────────────────────────────────────
# ARIMA ORDER SELECTION ONLY  (no forecast, no plots)
# ─────────────────────────────────────────────

if SELECT_ORDERS:
    print("\n" + "=" * 80)
    print(f"ARIMA ORDER SELECTION  (training data up to {year_n}  ·  criterion: AIC)")
    print("=" * 80)

    for name, df in assets.items():
        train = df[df["Date"] < f"{year_n}-01-01"].copy()
        best_order, grid = select_order(
            train["Log_Return"],
            train["Date"],
            criterion="aic",
        )
        print(f"\n  {name}")
        print(f"    Best order : ARIMA{best_order}")
        print(f"    Top 5 candidates:")
        print(grid.head(5).to_string(index=False))

# ─────────────────────────────────────────────
# FINAL ARIMA MODELS (from final_arima.py)
# ─────────────────────────────────────────────

if RUN_STATIC_ARIMA:
    run_arima_final("Static ARIMA  (no retraining)",
                    forecast_static, assets, year_n)

if RUN_LONGRUN_ARIMA:
    run_arima_final(f"Long-Run Static ARIMA  ({start_year}→{end_year})",
                    forecast_static_longrun, assets, year_n,
                    start_year=start_year, end_year=end_year, longrun=True)

if RUN_FINAL_ARIMA:
    run_arima_final("Final ARIMA  (expanding window)",
                    forecast_final, assets, year_n)

# ─────────────────────────────────────────────
# RANDOM WALK MODELS (from randomwalk.py)
# ─────────────────────────────────────────────

if RUN_RW_STATIC:
    run_randomwalk("Static Random Walk (no retraining)",
                   forecast_rw_static, assets, year_n)

if RUN_RW_MONTHLY:
    run_randomwalk("Monthly Random Walk  (expanding window)",
                   forecast_rw_monthly, assets, year_n)

if RUN_RW_LONGRUN:
    run_randomwalk(f"Long-Run Random Walk  ({start_year}→{end_year})",
                   forecast_rw_longrun, assets, year_n,
                   start_year=start_year, end_year=end_year, longrun=True)

# ─────────────────────────────────────────────
# XGBOOST MODELS (from boosting.py)
# ─────────────────────────────────────────────

if RUN_XGB_MONTHLY:
    run_xgb("XGBoost  (expanding window)",
            forecast_xgb_monthly, assets, year_n)

if RUN_XGB_STATIC:
    run_xgb("XGBoost  (static)",
            forecast_xgb_static, assets, year_n)

if RUN_XGB_LONGRUN:
    run_xgb(f"XGBoost  (long-run · {start_year}→{end_year})",
            forecast_xgb_longrun, assets, year_n,
            start_year=start_year, end_year=end_year, longrun=True)

# ─────────────────────────────────────────────
# ARCHIVED MODELS (from arima.py)
# ─────────────────────────────────────────────

if RUN_STATIC:
    run_arima_basic("Static",                         forecast_asset,                            assets, year_n)

if RUN_MONTHLY:
    run_arima_basic("Monthly",                        forecast_asset_monthly,                    assets, year_n)

if RUN_MONTHLY_PROP:
    run_arima_basic("Monthly Propagating",            forecast_asset_monthly_propagating,        assets, year_n)

if RUN_ROLLING:
    run_arima_basic("Rolling Window",                 forecast_asset_rolling_window,             assets, year_n)

if RUN_ROLLING_PROP:
    run_arima_basic("Rolling Window Propagating",     forecast_asset_rolling_window_propagating, assets, year_n)