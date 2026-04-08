from data import sp500, gold, eur_usd, bitcoin
from visual import plot_assets, plot_log_returns, print_quarterly_results, plot_forecast_with_train
from arima import forecast_asset, forecast_asset_monthly, forecast_asset_monthly_propagating

# ─────────────────────────────────────────────
# Group all assets into a single dict for convenience
# ─────────────────────────────────────────────

assets = {
    "Bitcoin": bitcoin,
    "SP500":   sp500,
    "Gold":    gold,
    "EUR/USD": eur_usd,
}


# ─────────────────────────────────────────────
# Visualizations — price charts and log returns
# ─────────────────────────────────────────────

#plot_assets(assets, mode="close")
#plot_assets(assets, mode="ma")
#plot_log_returns(assets)

# ─────────────────────────────────────────────
# ARIMA Forecasting and Evaluation
# ─────────────────────────────────────────────

year_n = 2020

for name, df in assets.items():
    train, forecast_df = forecast_asset(name, df, year_n)

    print_quarterly_results(name, forecast_df)
    plot_forecast_with_train(name, train, forecast_df)


for name, df in assets.items():
    train, forecast_df = forecast_asset_monthly(name, df, year_n)

    print_quarterly_results(name, forecast_df)
    plot_forecast_with_train(name, train, forecast_df)


for name, df in assets.items():
    train, forecast_df = forecast_asset_monthly_propagating(name, df, year_n)

    print_quarterly_results(name, forecast_df)
    plot_forecast_with_train(name, train, forecast_df)

