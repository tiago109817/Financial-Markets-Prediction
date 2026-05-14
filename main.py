from data import sp500, gold, eur_usd, bitcoin

from visual import (
    plot_assets, plot_log_returns,
    print_quarterly_results,
    plot_forecast_with_train,
)

from arima import (
    forecast_asset_monthly,
    # Archived models — uncomment here and set their flag to True to reactivate
    # forecast_asset,
    # forecast_asset_monthly_propagating,
    # forecast_asset_rolling_window,
    # forecast_asset_rolling_window_propagating,
)

from final_arima import (
    forecast_final,
    forecast_static,
    forecast_static_longrun,
    select_order,
)


# ─────────────────────────────────────────────
# CONFIGURATION FLAGS
# ─────────────────────────────────────────────

RUN_VISUALS        = False

SELECT_ORDERS      = False   # print best ARIMA order per asset (no forecast)
RUN_STATIC_ARIMA   = False   # one-shot forecast for a single year
RUN_LONGRUN_ARIMA  = True    # one-shot forecast across multiple years (2020→2026)
RUN_FINAL_ARIMA    = False   # monthly expanding-window, AIC/BIC order

RUN_MONTHLY        = False   # basic monthly (arima.py)
RUN_STATIC         = False   # archived
RUN_MONTHLY_PROP   = False   # archived
RUN_ROLLING        = False   # archived
RUN_ROLLING_PROP   = False   # archived

year_n     = 2024   # used by single-year models
start_year = 2020   # used by long-run model
end_year   = 2025   # used by long-run model


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
    plot_log_returns(assets)


# ─────────────────────────────────────────────
# ORDER SELECTION ONLY  (no forecast, no plots)
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
# STATIC ARIMA  (one-shot, single year)
# ─────────────────────────────────────────────

if RUN_STATIC_ARIMA:
    print("\n" + "=" * 80)
    print("MODEL: Static ARIMA  (one-shot · AIC/BIC order · no retraining)")
    print("=" * 80)

    for name, df in assets.items():
        train, forecast_df, best_order = forecast_static(
            name, df, year_n,
            criterion="aic",
        )
        print_quarterly_results(name, forecast_df)
        plot_forecast_with_train(
            f"{name}  —  Static ARIMA{best_order}",
            train,
            forecast_df,
        )


# ─────────────────────────────────────────────
# LONG-RUN STATIC ARIMA  (one-shot, 2020→2025)
# Fitted once on pre-2020 data, forecasts the
# entire multi-year horizon in a single call.
# No retraining, no re-anchoring — ever.
# ─────────────────────────────────────────────

if RUN_LONGRUN_ARIMA:
    print("\n" + "=" * 80)
    print(f"MODEL: Long-Run Static ARIMA  ({start_year}→{end_year}  ·  one-shot  ·  AIC/BIC order)")
    print("=" * 80)

    for name, df in assets.items():
        train, forecast_df, best_order = forecast_static_longrun(
            name, df, start_year, end_year,
            criterion="aic",
        )
        plot_forecast_with_train(
            f"{name}  —  Long-Run Static ARIMA{best_order}  ({start_year}→{end_year})",
            train,
            forecast_df,
        )


# ─────────────────────────────────────────────
# FINAL ARIMA MONTHLY (expanding-window)
# ─────────────────────────────────────────────

if RUN_FINAL_ARIMA:
    print("\n" + "=" * 80)
    print("MODEL: Final ARIMA  (expanding window · 1st-of-month anchor · AIC/BIC order)")
    print("=" * 80)

    for name, df in assets.items():
        train, forecast_df, best_order = forecast_final(
            name, df, year_n,
            criterion="aic",
        )
        print_quarterly_results(name, forecast_df)
        plot_forecast_with_train(
            f"{name}  —  Final ARIMA{best_order}",
            train,
            forecast_df,
        )


# ─────────────────────────────────────────────
# BASIC MONTHLY  (arima.py)
# ─────────────────────────────────────────────

if RUN_MONTHLY:
    print("\n" + "=" * 80)
    print("MODEL: Monthly  (expanding window · 1st-of-month anchor · fixed order)")
    print("=" * 80)

    for name, df in assets.items():
        train, forecast_df, best_order = forecast_asset_monthly(
            name, df, year_n,
            order=None,
            criterion="aic",
        )
        print_quarterly_results(name, forecast_df)
        plot_forecast_with_train(
            f"{name}  —  Monthly ARIMA{best_order}",
            train,
            forecast_df,
        )


# ─────────────────────────────────────────────
# ARCHIVED MODELS
# Set the flag to True and uncomment the import above to reactivate.
# ─────────────────────────────────────────────

# if RUN_STATIC:
#     run_model("Static", forecast_asset)

# if RUN_MONTHLY_PROP:
#     run_model("Monthly Propagating", forecast_asset_monthly_propagating)

# if RUN_ROLLING:
#     run_model("Rolling Window", forecast_asset_rolling_window)

# if RUN_ROLLING_PROP:
#     run_model("Rolling Window Propagating", forecast_asset_rolling_window_propagating)