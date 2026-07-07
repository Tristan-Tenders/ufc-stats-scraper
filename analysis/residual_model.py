"""Market-residual model: don't predict the winner — predict where the
MARKET is wrong.

The closing line already contains most knowable information. This trains a
regressor on (outcome - market_probability) using only stats features, then
blends: p = market + shrink * predicted_residual. If the blend can't beat the
raw market on log-loss/Brier out of time, the market is efficient against our
feature set — and that's the honest answer.

  python -m analysis.residual_model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss

from core.strategies import ODDS_COLS, mirror_rows, time_decay_weights
from core.utils import META_COLS, load_ufc_data, temporal_split


def american_profit(odds: float) -> float:
    return odds / 100 if odds > 0 else 100 / -odds


def main():
    df = pd.read_csv("stats/train_dataset.csv", parse_dates=["date"])
    X_df, y, dates = load_ufc_data()
    Xtr, Xte, ytr, yte = temporal_split(X_df, y, dates)

    # odds-covered subsets
    mtr = Xtr["a_implied_prob"].notna().to_numpy()
    mte = Xte["a_implied_prob"].notna().to_numpy()
    stats_cols = [c for c in X_df.columns if c not in ODDS_COLS]

    Xtr_o, ytr_o = Xtr[mtr], ytr[mtr]
    market_tr = Xtr_o["a_implied_prob"].to_numpy()
    resid_tr = ytr_o - market_tr

    Xte_o, yte_o = Xte[mte], yte[mte]
    market_te = Xte_o["a_implied_prob"].to_numpy()
    print(f"train fights with odds: {len(ytr_o)}   test: {len(yte_o)}")

    # regressor on the residual (stats features only, native NaN)
    w = time_decay_weights(dates[Xtr_o.index], 6.0)
    Xa, ra, wa = mirror_rows(Xtr_o, resid_tr + 0.0, w)  # mirroring flips resid
    # careful: mirror_rows flips labels via 1-y; for residuals flip sign instead
    ra = np.concatenate([resid_tr, -resid_tr])
    reg = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.04, max_iter=400, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=42)
    reg.fit(Xa[stats_cols].to_numpy(dtype=float), ra, sample_weight=wa)

    r_hat_te = reg.predict(Xte_o[stats_cols].to_numpy(dtype=float))

    # pick shrinkage on the last 20% of TRAIN (never test)
    n_val = max(int(len(Xtr_o) * 0.2), 100)
    Xval = Xtr_o.iloc[-n_val:]
    r_hat_val = reg.predict(Xval[stats_cols].to_numpy(dtype=float))
    mval = Xval["a_implied_prob"].to_numpy()
    yval = ytr_o[-n_val:]
    best_s, best_ll = 0.0, log_loss(yval, np.clip(mval, 0.02, 0.98))
    for s in (0.1, 0.25, 0.5, 0.75, 1.0):
        ll = log_loss(yval, np.clip(mval + s * r_hat_val, 0.02, 0.98))
        if ll < best_ll:
            best_s, best_ll = s, ll
    print(f"chosen shrinkage: {best_s} (validation logloss {best_ll:.4f})")

    blended = np.clip(market_te + best_s * r_hat_te, 0.02, 0.98)
    market_c = np.clip(market_te, 0.02, 0.98)

    print("\n=== out-of-time test (market-covered fights) ===")
    print(f"{'':14s} {'logloss':>8} {'brier':>8} {'acc':>7}")
    for name, p in (("market", market_c), ("blended", blended)):
        print(f"{name:14s} {log_loss(yte_o, p):8.4f} "
              f"{brier_score_loss(yte_o, p):8.4f} "
              f"{((p >= .5).astype(int) == yte_o).mean():7.3f}")
    verdict = "BLEND WINS — residual model found signal" \
        if log_loss(yte_o, blended) < log_loss(yte_o, market_c) \
        else "market wins — no exploitable residual signal in these features"
    print(f"-> {verdict}")

    # betting sim at real odds where the residual says the market is off
    meta = df.loc[Xte_o.index]
    has = meta["a_odds_am"].notna().to_numpy()
    print(f"\nflat 1u bets when |residual| >= thr ({has.sum()} priced fights):")
    print(f"  {'thr':>5} {'bets':>5} {'hit%':>6} {'profit':>8} {'roi':>7}")
    for thr in (0.02, 0.04, 0.06, 0.08, 0.12):
        mask = (np.abs(r_hat_te) >= thr) & has
        n = int(mask.sum())
        if n == 0:
            continue
        on_a = r_hat_te[mask] > 0
        won = np.where(on_a, yte_o[mask] == 1, yte_o[mask] == 0)
        pay = np.where(on_a,
                       [american_profit(o) for o in meta.loc[mask, "a_odds_am"]],
                       [american_profit(o) for o in meta.loc[mask, "b_odds_am"]])
        profit = float(np.where(won, pay, -1.0).sum())
        print(f"  {thr:5.2f} {n:5d} {won.mean() * 100:5.1f}% "
              f"{profit:+8.1f} {profit / n * 100:+6.1f}%")


if __name__ == "__main__":
    main()
