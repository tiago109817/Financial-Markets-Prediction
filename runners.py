from visual import (
    print_quarterly_results,
    plot_forecast_with_train,
    plot_rw_forecast,
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
        plot_forecast_with_train(f"{name}  —  {model_name}  ARIMA{best_order}", train, forecast_df)


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


# ─────────────────────────────────────────────
# Random Walk models (from randomwalk.py)
# ─────────────────────────────────────────────

def run_randomwalk(model_name, model_func, assets, year_n,
                   start_year=None, end_year=None, longrun=False):
    """
    Runner for all three random walk variants.

    Calls plot_rw_forecast instead of plot_forecast_with_train so that the
    ±1σ / ±2σ GBM confidence bands are rendered alongside the expected path.

    Parameters
    ----------
    model_name : str    Label printed in the section header and plot title.
    model_func : callable
        One of: forecast_rw_static, forecast_rw_monthly, forecast_rw_longrun.
    assets     : dict   {name: DataFrame}
    year_n     : int    Single-year target (used by static and monthly).
    start_year : int    Only used when longrun=True.
    end_year   : int    Only used when longrun=True.
    longrun    : bool   When True, calls model_func(name, df, start_year, end_year).
    """
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    for name, df in assets.items():

        if longrun:
            train, forecast_df, params = model_func(name, df, start_year, end_year)
        else:
            train, forecast_df, params = model_func(name, df, year_n)

        mu, sigma = params

        print_quarterly_results(name, forecast_df)
        plot_rw_forecast(
            f"{name}  —  {model_name}  (μ={mu}, σ={sigma})",
            train,
            forecast_df,
        )