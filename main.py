from data import sp500, gold, eur_usd, bitcoin
from visual import plot_close, plot_log_returns, plot_moving_average


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
# Visualizations
# ─────────────────────────────────────────────

plot_close(assets)
plot_log_returns(assets)
plot_moving_average(assets)