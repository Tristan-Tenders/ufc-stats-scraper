"""Round-by-round survival model: WHEN and HOW does the fight end?

Discrete-time hazard: for each round r, P(fight ends in r via KO / via SUB /
continues | it reached r). Chaining the hazards gives a full distribution
over (round, method) plus P(decision) — richer than the 3-class method model
and the basis for round/over-under props (the softest betting markets).

  python -m analysis.round_model            # train + temporal evaluation report
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from core.utils import META_COLS, temporal_split, load_ufc_data

CLASSES = ["continue", "ko", "sub"]


def scheduled_rounds(row) -> int:
    return 5 if row.get("is_title", 0) == 1 else 3


def build_survival_rows(df: pd.DataFrame, feature_cols: list[str]):
    """One row per (fight, round reached). Label: what happened IN that round."""
    ended_round = np.ceil(df["fight_secs"].fillna(900) / 300).clip(1, 5).astype(int)
    is_dec = (df["method"] == "dec").to_numpy()

    rows_X, rows_r, labels, fight_idx = [], [], [], []
    for i, (_, f) in enumerate(df.iterrows()):
        sched = scheduled_rounds(f)
        end_r = sched if is_dec[i] else int(ended_round.iloc[i])
        for r in range(1, min(end_r, sched) + 1):
            rows_X.append(i)
            rows_r.append(r)
            if r < end_r or is_dec[i]:
                labels.append("continue")
            else:
                labels.append(f["method"])  # 'ko' or 'sub' in the final round
            fight_idx.append(i)

    X = df.iloc[rows_X][feature_cols].reset_index(drop=True)
    X["round_num"] = rows_r
    return X, np.array(labels), np.array(fight_idx)


def fight_distribution(model, x_row: pd.DataFrame, sched: int = 3) -> dict:
    """Chain per-round hazards into P(ends round r via m) + P(decision)."""
    classes = list(model.classes_)
    out, p_alive = {}, 1.0
    for r in range(1, sched + 1):
        xr = x_row.copy()
        xr["round_num"] = r
        p = model.predict_proba(xr.to_numpy(dtype=float))[0]
        p_cont = p[classes.index("continue")]
        for m in ("ko", "sub"):
            out[f"r{r}_{m}"] = p_alive * p[classes.index(m)]
        p_alive *= p_cont
    out["decision"] = p_alive
    return out


def main():
    df = pd.read_csv("stats/train_dataset.csv", parse_dates=["date"])
    X_all, y, dates = load_ufc_data()
    feature_cols = X_all.columns.tolist()

    # temporal split at the fight level, then expand to rounds
    cutoff = dates.quantile(0.85)
    tr_mask = (dates < cutoff).to_numpy()
    print(f"cutoff {cutoff.date()}: {tr_mask.sum()} train / "
          f"{(~tr_mask).sum()} test fights")

    Xtr_r, ytr_r, _ = build_survival_rows(df[tr_mask], feature_cols)
    Xte_r, yte_r, te_fights = build_survival_rows(df[~tr_mask], feature_cols)
    print(f"round-rows: {len(ytr_r)} train / {len(yte_r)} test")

    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)
    model.fit(Xtr_r.to_numpy(dtype=float), ytr_r)

    # per-round hazard quality vs a constant-rate baseline
    proba = model.predict_proba(Xte_r.to_numpy(dtype=float))
    prior = np.tile(pd.get_dummies(pd.Series(ytr_r)).mean().reindex(
        model.classes_).to_numpy(), (len(yte_r), 1))
    print(f"\nper-round hazard logloss: model "
          f"{log_loss(yte_r, proba, labels=model.classes_):.4f}  vs "
          f"class-prior baseline "
          f"{log_loss(yte_r, prior, labels=model.classes_):.4f}")

    # fight-level: P(goes to decision) calibration in quintiles
    test_df = df[~tr_mask].reset_index(drop=True)
    p_dec = []
    for i in range(len(test_df)):
        x = test_df.iloc[[i]][feature_cols]
        dist = fight_distribution(model, x, scheduled_rounds(test_df.iloc[i]))
        p_dec.append(dist["decision"])
    p_dec = np.array(p_dec)
    actual_dec = (test_df["method"] == "dec").to_numpy()

    print(f"\nP(decision) calibration (test fights, actual decision rate "
          f"{actual_dec.mean():.1%}):")
    print(f"  {'pred bucket':>12} {'n':>5} {'predicted':>10} {'actual':>8}")
    q = pd.qcut(p_dec, 5, duplicates="drop")
    for bucket, grp in pd.DataFrame(
            {"p": p_dec, "y": actual_dec, "b": q}).groupby("b", observed=True):
        print(f"  {str(bucket):>12} {len(grp):5d} {grp['p'].mean():10.1%} "
              f"{grp['y'].mean():8.1%}")

    # over/under 1.5 rounds (ends in R1) quick check
    p_r1 = []
    for i in range(len(test_df)):
        x = test_df.iloc[[i]][feature_cols]
        d = fight_distribution(model, x, scheduled_rounds(test_df.iloc[i]))
        p_r1.append(d["r1_ko"] + d["r1_sub"])
    p_r1 = np.array(p_r1)
    ends_r1 = ((test_df["fight_secs"] <= 300) &
               (test_df["method"] != "dec")).to_numpy()
    print(f"\nends-in-R1: predicted mean {p_r1.mean():.1%} vs actual "
          f"{ends_r1.mean():.1%}  (logloss {log_loss(ends_r1, np.clip(p_r1, .01, .99)):.4f} "
          f"vs base-rate {log_loss(ends_r1, np.full_like(p_r1, ends_r1.mean())):.4f})")


if __name__ == "__main__":
    main()
