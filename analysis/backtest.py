"""Backtest the market-mismatch strategy on strictly future fights.

Fits the strategy ensemble on fights before the temporal cutoff, then on the
held-out recent fights: bets flat 1 unit whenever the model disagrees with the
vig-free market probability by more than a threshold, settled at the REAL
American odds (vig included). Reports ROI per threshold.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from core.strategies import StrategyEnsemble
from core.utils import META_COLS, load_ufc_data, temporal_split


def american_profit(odds: float) -> float:
    """Profit on a winning 1-unit bet."""
    return odds / 100 if odds > 0 else 100 / -odds


def main():
    df = pd.read_csv("stats/train_dataset.csv", parse_dates=["date"])
    X_df, y, dates = load_ufc_data()
    Xtr, Xte, ytr, yte = temporal_split(X_df, y, dates)

    print("fitting strategy ensemble (out-of-fold stacking)...")
    aux = df.loc[Xtr.index, ["sig_strike_diff", "fight_secs"]] \
        if "sig_strike_diff" in df.columns else None
    ens = StrategyEnsemble(X_df.columns.tolist()).fit(
        Xtr, ytr, dates=dates[Xtr.index], aux_targets=aux)

    print("\nmeta weights (how much each strategy is trusted):")
    for name, w in sorted(ens.weights.items(), key=lambda x: -abs(x[1])):
        print(f"  {name:20s} {w:+.2f}")

    p = ens.predict_proba(Xte)
    print("\nensemble on test (market-independent):")
    print(f"  acc: {accuracy_score(yte, (p >= .5).astype(int)):.3f}  "
          f"auc: {roc_auc_score(yte, p):.3f}  logloss: {log_loss(yte, p):.3f}")

    # market mismatch backtest on fights that have odds
    test_meta = df.loc[Xte.index]
    has = test_meta["a_odds_am"].notna() & Xte["a_implied_prob"].notna()
    market_p = Xte.loc[has, "a_implied_prob"].to_numpy()
    odds_a = test_meta.loc[has, "a_odds_am"].to_numpy()
    odds_b = test_meta.loc[has, "b_odds_am"].to_numpy()
    yy = yte[has.to_numpy()]
    pp = p[has.to_numpy()]
    edge = pp - market_p

    print(f"\nmarket comparison ({has.sum()} test fights with odds):")
    print(f"  market logloss: {log_loss(yy, market_p):.3f}   "
          f"model logloss: {log_loss(yy, pp):.3f}")

    print(f"\nflat 1-unit bets when |model - market| >= threshold "
          f"(settled at real odds, vig included):")
    print(f"  {'thr':>5} {'bets':>5} {'hit%':>6} {'profit':>8} {'roi':>7}")
    for thr in [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]:
        mask = np.abs(edge) >= thr
        n = int(mask.sum())
        if n == 0:
            continue
        bet_on_a = edge[mask] > 0
        won = np.where(bet_on_a, yy[mask] == 1, yy[mask] == 0)
        payout = np.where(bet_on_a,
                          [american_profit(o) for o in odds_a[mask]],
                          [american_profit(o) for o in odds_b[mask]])
        profit = float(np.where(won, payout, -1.0).sum())
        print(f"  {thr:5.2f} {n:5d} {won.mean() * 100:5.1f}% "
              f"{profit:+8.1f} {profit / n * 100:+6.1f}%")


if __name__ == "__main__":
    main()
