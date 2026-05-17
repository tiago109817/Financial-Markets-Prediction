from visual import (
    print_quarterly_results,
    plot_forecast_with_train,
)


# ─────────────────────────────────────────────
# Final ARIMA models (from final_arima.py)
# ─────────────────────────────────────────────

def run_arima_final(model_name, model_func, assets, year_n,
                    start_year=None, end_year=None, longrun=False):

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    for name, df in assets.items():

        if longrun:
            train, forecast_df, best_order = model_func(name, df, start_year, end_year, criterion="aic")
        else:
            train, forecast_df, best_order = model_func(name, df, year_n, criterion="aic")

        print_quarterly_results(name, forecast_df)
        plot_forecast_with_train(f"{name}  —  {model_name}{best_order}", train, forecast_df)


# ─────────────────────────────────────────────
# Basic ARIMA models (from arima.py)
# ─────────────────────────────────────────────

def run_arima_basic(model_name, model_func, assets, year_n):

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    for name, df in assets.items():

        data, forecast_df = model_func(name, df, year_n)

        print_quarterly_results(name, forecast_df)
        plot_forecast_with_train(f"{name} ({model_name})", data, forecast_df)