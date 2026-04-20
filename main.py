from data import sp500, gold, eur_usd, bitcoin
from visual import (
    plot_assets, plot_log_returns,
    print_quarterly_results,
    plot_forecast_with_train
)

from arima import (
    forecast_asset,
    forecast_asset_monthly,
    forecast_asset_monthly_propagating,
    forecast_asset_rolling_window,
    forecast_asset_rolling_window_propagating
)

# ─────────────────────────────────────────────
# CONFIGURATION FLAGS
# ─────────────────────────────────────────────

RUN_VISUALS = False
RUN_STATIC = True
RUN_MONTHLY = True
RUN_MONTHLY_PROP = True
RUN_ROLLING = True
RUN_ROLLING_PROP = True

year_n = 2024

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
# Helper runner (avoids repetition)
# ─────────────────────────────────────────────

def run_model(model_name, model_func, use_full_df=False):

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    for name, df in assets.items():

        data, forecast_df = model_func(name, df, year_n)

        print_quarterly_results(name, forecast_df)

        plot_df = data  # train or full depending on model

        plot_forecast_with_train(
            f"{name} ({model_name})",
            plot_df,
            forecast_df
        )


# ─────────────────────────────────────────────
# MODELS (toggle individually)
# ─────────────────────────────────────────────

if RUN_STATIC:
    run_model("Static", forecast_asset)

if RUN_MONTHLY:
    run_model("Monthly", forecast_asset_monthly)

if RUN_MONTHLY_PROP:
    run_model("Monthly Propagating", forecast_asset_monthly_propagating)

if RUN_ROLLING:
    run_model("Rolling Window", forecast_asset_rolling_window)

if RUN_ROLLING_PROP:
    run_model("Rolling Window Propagating", forecast_asset_rolling_window_propagating)
