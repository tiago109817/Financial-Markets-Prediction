# Financial Markets Prediction

BSc Thesis comparing ARIMA, XGBoost, and Random Walk at forecasting S&P 500, Gold, EUR/USD, and Bitcoin.

**`main.py`** — Entry point for all experiments. Toggle boolean flags to select models and set `year_n` or `start_year`/`end_year` for the forecast horizon.

**`data.py`** — Loads and preprocesses the Excel files into a continuous daily time series. Computes log returns and a 13-day moving average, and forward-fills weekends and holidays.

**`final_arima.py`** — Production ARIMA implementation with automatic order selection via AIC/BIC grid search. Contains three variants: monthly expanding window, static one-shot, and long-run multi-year.

**`arima.py`** — Archived early ARIMA prototypes with a fixed (2,0,2) order. Kept for reference; includes static, monthly, propagating, and rolling-window variants.

**`randomwalk.py`** — GBM random walk estimating daily drift (μ) and volatility (σ) from historical log returns. Produces the deterministic expected path plus ±1σ and ±2σ confidence bands.

**`boosting.py`** — XGBoost model using lagged returns and rolling volatility as features. Hyperparameters are selected via time-series cross-validation, and forecasts are generated recursively step by step.

**`rnn.py`** — RNN/LSTM model (in development).

**`runners.py`** — Helper functions that handle printing quarterly results and dispatching the correct plot for each model family.

**`visual.py`** — All plots: raw prices, log returns, and forecast charts with optional GBM confidence bands for the random walk.
