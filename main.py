from data import sp500, gold, eur_usd, bitcoin
from visual import plot_assets, plot_log_returns


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

plot_assets(assets, mode="close")
plot_assets(assets, mode="ma")
plot_log_returns(assets)