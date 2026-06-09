import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
#
# Neural networks rely on randomness while they learn (random starting values,
# random shuffling of the data, and some randomness on GPUs). This means that
# running the exact same model twice would normally give slightly different
# results each time.
#
# To avoid that, we fix this randomness to a single starting point (the "seed").
# With it pinned, the model produces the SAME forecast every time it runs — which
# is essential so the results are stable and can be fairly compared with the
# other models (ARIMA, random walk, XGBoost). The number 42 is just an arbitrary
# fixed choice; any number would do.

# DEVICE is used to select NVIDIA's GPU if available, which can speed up training significantly.  
# If no GPU is available, it defaults to the CPU.

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed=SEED):
    """Pin numpy + torch RNGs and disable non-deterministic cuDNN kernels."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (single source of truth — used in train AND inference)
# ──────────────────────────────────────────────────────────────────────────────
#
# Unlike boosting.py we do NOT build explicit lag_{n} columns: the LSTM receives
# a sequence of consecutive days, so the lags are implicit in the time axis.
# What we DO add are per-day context features (volatility, momentum, distance
# from the moving average, calendar flags) that give each timestep richer
# information than the bare return alone.
#
# Crucially this is the ONLY place features are defined, and it is recomputable
# from a rolling (Date, Close, Log_Return) buffer — exactly like _add_features
# in boosting.py — so the recursive forecaster feeds the model inputs that are
# computed identically to training (no train/inference skew).

VOL_WINDOWS = [5, 10, 21]     # rolling-std windows (one week, one month)
MOM_WINDOWS = [5, 10, 21]     # rolling-sum windows — trend direction
MA_WINDOWS  = [10, 21]        # moving-average window for the mean-reversion gap
_EPS        = 1e-8            # guard against divide-by-zero


def _add_features(df):
    """
    Compute every per-timestep feature from a (Date, Close, Log_Return) frame.

    Features (all strictly backward-looking — no leakage)
    -----------------------------------------------------
    ret          the raw daily log return (the main signal).
    vol_{w}      rolling std of returns over w days — volatility clusters in
                 markets, so this flags the *size* of likely moves.
    mom_{w}      rolling SUM of returns over w days = log(P_t / P_{t-w}) —
                 captures trend DIRECTION, which volatility alone cannot.
    vol_ratio    vol_5 / vol_21 — >1 means short-term vol is elevated vs the
                 monthly baseline, a simple regime flag.
    ma_gap_21    log(Close / 21-day mean) — how far price sits from its own
                 recent average, a mean-reversion signal.
    dow          day-of-week (0=Mon … 6=Sun) — captures weak calendar effects.
    is_filled    1 on weekends (forward-filled rows in data.py), else 0.

    Target
    ------
    next_return  Log_Return shifted by -1 — the value the model predicts.

    NaN rows (from rolling windows / the final shift) are NOT dropped here;
    callers decide, because the recursive forecaster only reads the final row.
    """
    f = df.copy()
    f["Date"] = pd.to_datetime(f["Date"])

    f["ret"] = f["Log_Return"]

    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = f["Log_Return"].rolling(w).std()
    for w in MOM_WINDOWS:
        f[f"mom_{w}"] = f["Log_Return"].rolling(w).sum()

    f["vol_ratio"] = f["vol_5"] / (f["vol_21"] + _EPS)

    for w in MA_WINDOWS:
        ma = f["Close"].rolling(w).mean()
        f[f"ma_gap_{w}"] = np.log(f["Close"] / ma)

    f["dow"]       = f["Date"].dt.dayofweek
    f["is_filled"] = (f["Date"].dt.dayofweek >= 5).astype(int)

    f["next_return"] = f["Log_Return"].shift(-1)
    return f

FEATURE_COLS = (
    ["ret"]
    + [f"vol_{w}" for w in VOL_WINDOWS]
    + [f"mom_{w}" for w in MOM_WINDOWS]
    + ["vol_ratio"]
    + [f"ma_gap_{w}" for w in MA_WINDOWS]
    + ["dow", "is_filled"]
)
TARGET_COL = "next_return"


def _training_frame(df):
    """Full feature frame with all NaN feature/target rows removed."""
    f = _add_features(df)
    return f.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# STANDARDISATION  (fit on TRAINING data only — no leakage)
# ──────────────────────────────────────────────────────────────────────────────
#
# WHY:  A neural net learns badly when its input numbers live on very different
#       scales (daily returns are ~0.001, day-of-week is 0–6, etc.) — it mistakes
#       "bigger number" for "more important". So we rescale every column to the
#       same scale: subtract its average and divide by its spread (this is called
#       z-scoring). Afterwards each column is centered on 0 with a typical range
#       of about ±1, so the network compares them fairly.
#
# HOW (no leakage): we measure the average and spread on the TRAINING data only,
#       store those numbers, and reuse the exact same ones everywhere. If we used
#       the future (test) data to measure them, the model would secretly "see"
#       the future — that's cheating (data leakage) and makes results look better
#       than they really are. Predictions come out rescaled and are converted
#       back to a real return using the same stored numbers in reverse.

def _fit_scalers(train_feat):
    """
    Measure how to rescale the data, using the TRAINING rows only.

    Returns four numbers we keep and reuse everywhere:
        feat_mu, feat_sd : the average and spread of each FEATURE column
        tgt_mu,  tgt_sd  : the average and spread of the TARGET (the return
                           we want to predict)
    The tiny + _EPS just avoids ever dividing by zero on a flat column.
    """
    X = train_feat[FEATURE_COLS].to_numpy(dtype=float)
    y = train_feat[TARGET_COL].to_numpy(dtype=float)
    feat_mu = X.mean(axis=0)
    feat_sd = X.std(axis=0) + _EPS
    tgt_mu  = float(y.mean())
    tgt_sd  = float(y.std() + _EPS)
    return feat_mu, feat_sd, tgt_mu, tgt_sd


def _make_windows(feat_df, seq_len, feat_mu, feat_sd, tgt_mu, tgt_sd):
    """
    Slice the time series into the short sequences an LSTM learns from.

    WHY:  An LSTM doesn't look at one day at a time — it reads a STRETCH of
          consecutive days and predicts what comes next. `seq_len` is how many
          days are in that stretch (e.g. 20).

    HOW:  First we rescale the features (X) and the target (y) with the numbers
          from _fit_scalers. Then we slide a window across the data: for each
          position i we take the block of days [i-seq_len+1 … i] as the INPUT,
          and the return on the NEXT day as the ANSWER. Sliding forward one day
          at a time gives thousands of overlapping training examples that all
          say: "given these last `seq_len` days, predict tomorrow."

    No leakage: a window only ever contains days up to day i, and we ask it to
    predict day i+1 — the answer is never inside the window the model sees.

    Returns
    -------
    X : array, shape (number of windows, seq_len, number of features), float32
    y : array, shape (number of windows,),                             float32
    """
    X = (feat_df[FEATURE_COLS].to_numpy(dtype=float) - feat_mu) / feat_sd
    y = (feat_df[TARGET_COL].to_numpy(dtype=float)   - tgt_mu)  / tgt_sd

    seqs, tgts = [], []
    for i in range(seq_len - 1, len(X)):
        seqs.append(X[i - seq_len + 1 : i + 1])   # input: this block of days
        tgts.append(y[i])                         # answer: the next day's return

    return np.asarray(seqs, dtype=np.float32), np.asarray(tgts, dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL  (the neural network itself)
# ──────────────────────────────────────────────────────────────────────────────
#
# This class is the blueprint of the network. It's made of two parts:
#   1. an LSTM layer that reads a stretch of days one by one, keeping a running
#      "memory" of what it has seen so far, and
#   2. a final layer ("head") that turns that memory into ONE number — the
#      predicted next-day return.
#
# So the flow is:  a sequence of days  →  LSTM (summarises it)  →  one prediction.

class _LSTMForecaster(nn.Module):
    """
    The forecasting network.

    Build-time settings (the knobs the hyperparameter search tunes):
        n_features   how many numbers describe each day (our feature columns)
        hidden_size  how big the LSTM's memory is — bigger = more capacity
        num_layers   how many LSTM layers are stacked (1 = simple, 2 = deeper)
        dropout      a safeguard that randomly hides part of the signal while
                     training, so the model learns real patterns instead of
                     memorising noise (only used when there are 2+ layers)
    """

    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        # The "reader": processes the day-by-day sequence and remembers context.
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # The "decision": turns the LSTM's summary into a single predicted return.
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):                  # x = a batch of day-sequences
        # Let the LSTM read the whole sequence; it reports its memory at every day.
        out, _ = self.lstm(x)
        # Keep only the LAST day's memory — it has seen the entire window.
        last = out[:, -1, :]
        # Turn that into one prediction per sequence (and tidy the shape).
        return self.head(last).squeeze(-1)

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING  (with early stopping)
# ──────────────────────────────────────────────────────────────────────────────
#
# WHY:  This is where the network actually learns. It looks at the training
#       examples over and over, each time adjusting its internal settings to make
#       its predictions a little less wrong. We measure "wrongness" with Huber
#       loss, which stays precise on normal days but refuses to be dominated by
#       rare extreme days (crashes/rallies) — the right choice for fat-tailed
#       financial returns.
#
# HOW:  We don't just train forever. We set aside a recent slice of data the model
#       never trains on (the "validation" set) and, after each pass, check how it
#       does there. As long as it keeps improving we continue; once it stops
#       improving for several passes in a row we stop early and keep the best
#       version. This prevents the model from over-memorising the training data —
#       the neural-net version of the complexity penalty (AIC/BIC) in the ARIMA models.

def _train(model, X_tr, y_tr, X_val, y_val, config, device):
    """
    Teach the model on (X_tr, y_tr), judging its progress on (X_val, y_val),
    and return the best validation score it reached.
    """
    # Adam = the routine that decides how to adjust the model's settings each step.
    opt     = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.SmoothL1Loss()      # Huber loss (see note above)

    # Move the data into PyTorch's number format, on the chosen hardware.
    Xtr = torch.tensor(X_tr, device=device)
    ytr = torch.tensor(y_tr, device=device)
    Xva = torch.tensor(X_val, device=device)
    yva = torch.tensor(y_val, device=device)

    n_train  = Xtr.shape[0]
    batch    = config["batch_size"]      # how many examples to learn from at once
    patience = config["patience"]        # how many no-improvement passes we tolerate

    best_val   = float("inf")            # best validation score so far (lower = better)
    best_state = None                    # a saved copy of the best version of the model
    bad_epochs = 0                       # how many passes in a row haven't improved

    # One loop = one "epoch" = one full pass over all the training data.
    for _ in range(config["max_epochs"]):

        # ── learn from the training data ─────────────────────────────────────
        model.train()
        # Shuffle the order so the model doesn't learn the calendar order by accident,
        # then feed the data in small batches, improving a little after each batch.
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()                          # reset last step's adjustments
            loss = loss_fn(model(Xtr[idx]), ytr[idx])# how wrong on this batch
            loss.backward()                          # work out how to fix it
            opt.step()                               # apply the fix

        # ── check it on data it did NOT train on ─────────────────────────────
        model.eval()
        with torch.no_grad():                        # just checking, not learning
            val_mae = torch.mean(torch.abs(model(Xva) - yva)).item()

        # ── early stopping ───────────────────────────────────────────────────
        if val_mae < best_val - 1e-7:                # improved → save this version
            best_val   = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:                                        # no improvement → count it
            bad_epochs += 1
            if bad_epochs >= patience:               # too many in a row → stop early
                break

    # Roll back to the best version we saw, not necessarily the last one.
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val


# ──────────────────────────────────────────────────────────────────────────────
# FIT  (everything needed to train ONE version of the model)
# ──────────────────────────────────────────────────────────────────────────────
#
# WHY:  Training needs more than just the network — it needs the data rescaled,
#       chopped into sequences, and split so we can tell whether the model is
#       actually learning or just memorising. This function does all of that for
#       one set of settings and hands back a ready-to-use, trained model.
#
# HOW:  We hold back the most RECENT slice of training data (the last 15%) as a
#       "validation" set the model never learns from. Because we're forecasting
#       the future, those recent days are the closest stand-in for the future, so
#       checking the model on them is a fair preview of how it will really do.

VAL_FRAC = 0.15   # keep the last 15% of the data aside to judge the model


def _fit_with_val(train_feat, config, device):
    """
    Train one configuration of the model and report how well it did.

    Steps:
      1. Pin randomness so the result is repeatable.
      2. Work out how to rescale the data, and chop it into day-sequences.
      3. Split off the most recent 15% as a check ("validation") set.
      4. Build and train the network on the rest.
      5. Pack the trained model together with the rescaling numbers it needs
         (a "bundle") and return it, plus its score.

    The score is the average prediction error on the validation set, converted
    back to REAL return units so it can be compared fairly across settings and
    across assets.
    """
    _seed_everything()

    # How to rescale the numbers, and the data turned into day-sequences.
    feat_mu, feat_sd, tgt_mu, tgt_sd = _fit_scalers(train_feat)
    X, y = _make_windows(train_feat, config["seq_len"],
                         feat_mu, feat_sd, tgt_mu, tgt_sd)

    # Safety check: refuse to train if there's barely any data.
    if len(X) < 50:
        raise ValueError("Not enough windows to train the LSTM.")

    # Split in time order: first 85% to learn from, last 15% to check on.
    split = max(int(len(X) * (1 - VAL_FRAC)), 1)
    X_tr, y_tr   = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    if len(X_val) == 0:                       # tiny-data safeguard: reuse last row
        X_val, y_val = X_tr[-1:], y_tr[-1:]

    # Build the network with these settings.
    model = _LSTMForecaster(
        n_features  = len(FEATURE_COLS),
        hidden_size = config["hidden_size"],
        num_layers  = config["num_layers"],
        dropout     = config["dropout"],
    ).to(device)

    # Train it, then convert the score from rescaled units back to real returns
    # (multiplying by tgt_sd undoes the earlier division — so the error becomes
    # a real, readable number again).
    val_mae_scaled = _train(model, X_tr, y_tr, X_val, y_val, config, device)
    val_mae_raw    = val_mae_scaled * tgt_sd

    # Bundle the trained model with the exact rescaling numbers it was trained
    # with, so later predictions use the SAME settings and nothing gets mixed up.
    bundle = {
        "model":   model,
        "feat_mu": feat_mu, "feat_sd": feat_sd,
        "tgt_mu":  tgt_mu,  "tgt_sd":  tgt_sd,
        "seq_len": config["seq_len"],
        "device":  device,
    }
    return bundle, val_mae_raw


def _fit(train_feat, config, device):
    """
    Shortcut for when we only want the trained model and don't need its score —
    it just runs _fit_with_val and drops the score.
    """
    bundle, _ = _fit_with_val(train_feat, config, device)
    return bundle


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SELECTION  (small grid, chronological validation)
# ──────────────────────────────────────────────────────────────────────────────
#
# The grid is deliberately compact.  Each candidate trains a full LSTM, and the
# monthly model retrains every month, so an exhaustive search (as in ARIMA's
# (p,d,q) grid or XGBoost's 5-fold CV) would be prohibitively slow.  These four
# knobs are the ones that matter most for an LSTM on daily returns:
#
#   seq_len        how many past days the network sees per prediction (memory).
#   hidden_size    capacity of the recurrent state.
#   num_layers     1 (shallow) vs 2 (stacked, deeper temporal abstraction).
#   learning_rate  Adam step size.
#
# Everything else is fixed at sensible, regularised defaults.

SEQ_LENS       = [20, 40]
HIDDEN_SIZES   = [32, 64]
NUM_LAYERS     = [1, 2, 3]
LEARNING_RATES = [1e-3]

_FIXED = {
    "dropout":    0.2,    # active only when num_layers > 1
    "batch_size": 64,
    "max_epochs": 80,
    "patience":   10,
}


def _config_grid():
    """Cartesian product of the tunable knobs, each merged with _FIXED."""
    grid = []
    for seq in SEQ_LENS:
        for hidden in HIDDEN_SIZES:
            for layers in NUM_LAYERS:
                for lr in LEARNING_RATES:
                    cfg = {
                        "seq_len":       seq,
                        "hidden_size":   hidden,
                        "num_layers":    layers,
                        "learning_rate": lr,
                    }
                    cfg.update(_FIXED)
                    grid.append(cfg)
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG SELECTION  (try several setups, keep the best)
# ──────────────────────────────────────────────────────────────────────────────

def _select_config(train_feat, device):
    """
    Train every candidate setup and keep the one that predicts best.

    Returns
    -------
    best_config : dict         The winning settings.
    results_df  : table        Every setup tried, sorted best-first.
    best_bundle : trained model The winner, ready to forecast with.
    """
    records     = []                 # one row per setup, for the results table
    best_config = None               # the winning settings so far
    best_bundle = None               # the winning trained model so far
    best_mae    = float("inf")       # the winning (lowest) error so far

    # Go through each candidate setup in the list.
    for cfg in _config_grid():
        try:
            # Train it and get its error on the recent check data.
            bundle, val_mae = _fit_with_val(train_feat, cfg, device)
        except ValueError:
            # Not enough data for this setup — skip it instead of crashing.
            continue

        # Remember this setup and its score for the results table.
        records.append({
            "seq_len":       cfg["seq_len"],
            "hidden_size":   cfg["hidden_size"],
            "num_layers":    cfg["num_layers"],
            "learning_rate": cfg["learning_rate"],
            "val_mae":       val_mae,
        })

        # If it's the best so far, hold on to it.
        if val_mae < best_mae:
            best_mae    = val_mae
            best_config = cfg
            best_bundle = bundle

    # Build the results table, sorted from best (lowest error) to worst.
    results_df = pd.DataFrame(records).sort_values("val_mae").reset_index(drop=True)
    return best_config, results_df, best_bundle


def _print_config(name, best_config, results_df):
    """Print the chosen configuration + ranked grid (mirrors ARIMA/XGB output)."""
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  LSTM hyperparameter selection")
    print(f"{'=' * 60}")
    print(f"  Best config : seq_len={best_config['seq_len']}, "
          f"hidden_size={best_config['hidden_size']}, "
          f"num_layers={best_config['num_layers']}, "
          f"learning_rate={best_config['learning_rate']}")
    print(f"  Top 5 candidates (by validation MAE):")
    print(results_df.head(5).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# BUILDING THE FORECAST  (predict one day, then chain forward)
# ──────────────────────────────────────────────────────────────────────────────
#
# WHY:  The model only ever predicts ONE day ahead (tomorrow's return). But we
#       want a whole year. So we chain: predict tomorrow, then PRETEND that guess
#       really happened, add it to the data, and use it to predict the day after —
#       and so on, day by day, to the end. This is what "recursive" means.
#
#       Side effect to be aware of: because each guess is built on top of earlier
#       guesses, small errors can pile up the further out we go. That's exactly
#       why the one-shot static / long-run versions drift away from reality over
#       time, while the monthly version (which restarts from the real price each
#       month) stays much closer.
#
# HOW:  We keep a small running record of recent days ("buffer"). For each future
#       day we rebuild the features from that buffer (using the SAME recipe as
#       training), feed the most recent stretch into the model, turn its answer
#       back into a real return and then into a price, and tack that new day onto
#       the buffer before moving on.

def _recursive_path(bundle, history, future_dates, anchor_price):
    """
    Build a day-by-day price forecast, using each prediction to make the next.

    Parameters
    ----------
    bundle       : the trained model plus the rescaling numbers it needs.
    history      : the most recent REAL days, used to get the chain started.
    future_dates : the list of days we want to forecast.
    anchor_price : the last known REAL price — where the forecast path begins.

    Returns
    -------
    A list of forecast prices, one for each future date.
    """
    # Unpack the trained model and the exact rescaling numbers it was trained with.
    model   = bundle["model"]
    seq_len = bundle["seq_len"]
    device  = bundle["device"]
    fmu, fsd = bundle["feat_mu"], bundle["feat_sd"]
    tmu, tsd = bundle["tgt_mu"],  bundle["tgt_sd"]

    model.eval()   # prediction mode (turns off training-only behaviour)

    # Start from a copy of the recent real days.
    buf = history[["Date", "Close", "Log_Return"]].copy()
    buf["Date"] = pd.to_datetime(buf["Date"])

    prev_price = anchor_price       # the price we build each new day from
    prices     = []                 # the forecast we're filling in
    cap        = seq_len + 80       # keep the running record short, for speed

    # Walk forward one future day at a time.
    for d in pd.to_datetime(pd.Series(list(future_dates))).tolist():

        # 1. Rebuild the features and take the most recent stretch of days.
        feats  = _add_features(buf)
        window = feats[FEATURE_COLS].to_numpy(dtype=float)[-seq_len:]

        # 2. Rescale it the same way the model was trained on (and fill any gaps).
        window = (window - fmu) / fsd
        window = np.nan_to_num(window, nan=0.0)   # 0 = the average, after rescaling

        # 3. Ask the model for tomorrow's return, then undo the rescaling.
        x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            z = float(model(x).item())
        r = z * tsd + tmu                           # back to a real return

        # 4. Turn the return into a price and record it.
        new_price = prev_price * np.exp(r)
        prices.append(new_price)

        # 5. Pretend this predicted day really happened: add it to the record so
        #    the next prediction can build on it.
        buf = pd.concat(
            [buf, pd.DataFrame({"Date": [d], "Close": [new_price], "Log_Return": [r]})],
            ignore_index=True,
        )
        if len(buf) > cap:                          # trim old days to stay fast
            buf = buf.iloc[-cap:].reset_index(drop=True)

        prev_price = new_price                      # move forward

    return prices


# ──────────────────────────────────────────────────────────────────────────────
# CORE ENGINE  —  monthly expanding window
# ──────────────────────────────────────────────────────────────────────────────

def _run(df, full_features, start, end, config, device):
    """
    Monthly expanding-window LSTM engine.  Mirrors _run() in boosting.py /
    final_arima.py exactly:

      For each calendar month in [start, end]:
        1. Retrain the LSTM on all feature rows whose Date < current month.
        2. Anchor the path at the real close on the 1st of the current month
           (last available close handles weekends / holidays).
        3. Recursively forecast next-day returns for the whole month and
           convert them to prices from the anchor.
        4. Append results, advance to the next month.

    Month boundaries:  Jan 1 (anchor) → Feb 1 (last forecast day, inclusive).

    Returns
    -------
    pd.DataFrame   Columns: Date | Forecast | Real
    """
    all_forecasts = []
    current_date  = start

    while current_date <= end:

        next_month = current_date + pd.offsets.MonthBegin(1)

        test_slice = df[
            (df["Date"] >= current_date) &
            (df["Date"] <= next_month)
        ].copy()

        train_feat = full_features[full_features["Date"] < current_date].copy()

        if test_slice.empty or len(train_feat) < 50:
            current_date = next_month
            continue

        # Anchor: real close on / before the 1st of the current month.
        anchor_rows  = df[df["Date"] <= current_date]
        anchor_price = (
            anchor_rows["Close"].iloc[-1]
            if not anchor_rows.empty
            else train_feat["Close"].iloc[-1]
        )

        bundle = _fit(train_feat, config, device)

        # Seed the recursion with real data through the anchor day.
        history = anchor_rows[["Date", "Close", "Log_Return"]].tail(config["seq_len"] + 60)

        forecast_prices = _recursive_path(
            bundle, history, test_slice["Date"].values, anchor_price
        )

        all_forecasts.append(pd.DataFrame({
            "Date":     test_slice["Date"].values,
            "Forecast": forecast_prices,
            "Real":     test_slice["Close"].values,
        }))

        current_date = next_month

    return pd.concat(all_forecasts, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def forecast_rnn_monthly(name, df, year_n):
    """
    Monthly expanding-window LSTM for a single asset.  Mirrors
    forecast_final() (ARIMA) and forecast_xgb_monthly() (XGBoost): the network
    is retrained every month on a growing training set and re-anchored to the
    real close at the start of each month.

    The configuration is selected ONCE on the pre-year_n data, then reused for
    every monthly refit — the same discipline boosting.py applies to its CV grid.

    Returns
    -------
    train       : DataFrame   Historical data up to start of year_n.
    forecast_df : DataFrame   Date | Forecast | Real  for the full year.
    best_config : dict        Hyperparameters chosen on validation MAE.
    """
    _seed_everything()

    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n}-12-01")   # last iteration: Dec 1 → Jan 1
    train = df[df["Date"] < start].copy()

    full_features = _training_frame(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    best_config, results_df, _ = _select_config(train_feat, DEVICE)
    _print_config(name, best_config, results_df)

    forecast_df = _run(df, full_features, start, end, best_config, DEVICE)

    return train, forecast_df, best_config


def forecast_rnn_static(name, df, year_n):
    """
    Static (one-shot) LSTM forecast.  Mirrors forecast_static() (ARIMA) and
    forecast_xgb_static() (XGBoost): fitted ONCE on all data before year_n, then
    the full year is forecast recursively in a single pass — errors accumulate
    freely, which is an honest test of medium-term skill.

    The winning model from selection is reused directly (no extra refit).

    Returns
    -------
    train, forecast_df (Date | Forecast | Real), best_config
    """
    _seed_everything()

    start = pd.Timestamp(f"{year_n}-01-01")
    end   = pd.Timestamp(f"{year_n + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    full_features = _training_frame(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    best_config, results_df, best_bundle = _select_config(train_feat, DEVICE)
    _print_config(name, best_config, results_df)

    anchor_price    = train["Close"].iloc[-1]
    history         = train[["Date", "Close", "Log_Return"]].tail(best_config["seq_len"] + 60)
    forecast_prices = _recursive_path(best_bundle, history, test["Date"].values, anchor_price)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
    })

    return train, forecast_df, best_config


def forecast_rnn_longrun(name, df, start_year, end_year):
    """
    Long-run static LSTM forecast across multiple years.  Mirrors
    forecast_static_longrun() (ARIMA) and forecast_xgb_longrun() (XGBoost):
    fitted ONCE on all data before start_year, then every day through end_year
    is forecast recursively with no retraining and no re-anchoring — the most
    demanding test of long-term skill.

    Returns
    -------
    train, forecast_df (Date | Forecast | Real), best_config
    """
    _seed_everything()

    start = pd.Timestamp(f"{start_year}-01-01")
    end   = pd.Timestamp(f"{end_year + 1}-01-01")
    train = df[df["Date"] < start].copy()
    test  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    full_features = _training_frame(df)
    train_feat    = full_features[full_features["Date"] < start].copy()

    print(f"\n{'=' * 60}")
    print(f"  {name}  —  LSTM hyperparameter selection")
    print(f"  Forecast horizon: {start_year} → {end_year}")
    print(f"{'=' * 60}")

    best_config, results_df, best_bundle = _select_config(train_feat, DEVICE)
    print(f"  Best config : seq_len={best_config['seq_len']}, "
          f"hidden_size={best_config['hidden_size']}, "
          f"num_layers={best_config['num_layers']}, "
          f"learning_rate={best_config['learning_rate']}")
    print(f"  Top 5 candidates (by validation MAE):")
    print(results_df.head(5).to_string(index=False))

    anchor_price    = train["Close"].iloc[-1]
    history         = train[["Date", "Close", "Log_Return"]].tail(best_config["seq_len"] + 60)
    forecast_prices = _recursive_path(best_bundle, history, test["Date"].values, anchor_price)

    forecast_df = pd.DataFrame({
        "Date":     test["Date"].values,
        "Forecast": forecast_prices,
        "Real":     test["Close"].values,
    })

    # ── Final day comparison ──────────────────────────────────────────────────
    last  = forecast_df.iloc[-1]
    error = ((last["Forecast"] - last["Real"]) / last["Real"]) * 100
    print(f"\n  Final day  :  {last['Date'].date()}")
    if name == "EUR/USD":
        print(f"  Forecast   :  {last['Forecast']:.4f}")
        print(f"  Real       :  {last['Real']:.4f}")
    else:
        print(f"  Forecast   :  {last['Forecast']:.2f}")
        print(f"  Real       :  {last['Real']:.2f}")
    print(f"  Error      :  {error:.2f}%")

    return train, forecast_df, best_config