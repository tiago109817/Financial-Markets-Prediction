from visual import (
    print_quarterly_results,
    plot_forecast_with_train,
    plot_rw_forecast,
    plot_rw_forecast_zoom,
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

        # For the monthly model mu and sigma are re-estimated each month, so
        # the returned params are only the initial values — showing them in the
        # title would be misleading. For static and long-run a single pair is
        # used throughout, so it is meaningful to display.
        if model_func.__name__ == "forecast_rw_monthly":
            title = f"{name}  —  {model_name}"
        else:
            title = f"{name}  —  {model_name}  (μ={mu:.6f}, σ={sigma:.6f})"

        if not longrun:
            print_quarterly_results(name, forecast_df)
            
        plot_rw_forecast(title, train, forecast_df)
        plot_rw_forecast_zoom(title, forecast_df)
            
# ─────────────────────────────────────────────
# XGBoost models (from boosting.py)
# ─────────────────────────────────────────────
 
def run_xgb(model_name, model_func, assets, year_n,
            start_year=None, end_year=None, longrun=False):
    """
    Runner for all three XGBoost variants.
 
    Return signature of all three boosting functions:
        (train, forecast_df, best_params)
 
    Uses plot_forecast_with_train — same as ARIMA — since there are no
    confidence bands to add.  The best_params dict is printed by the model
    function itself; here we just show the chosen params in the plot title.
    """
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)
 
    for name, df in assets.items():
 
        if longrun:
            train, forecast_df, best_params = model_func(name, df, start_year, end_year)
        else:
            train, forecast_df, best_params = model_func(name, df, year_n)
 
        params_str = (f"n_est={best_params['n_estimators']}  "
                      f"depth={best_params['max_depth']}  "
                      f"lr={best_params['learning_rate']}")
 
        print_quarterly_results(name, forecast_df)
        plot_forecast_with_train(
            f"{name}  —  {model_name}  ({params_str})",
            train,
            forecast_df,
        )