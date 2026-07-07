"""Train and evaluate fight-outcome models on real fight data.

Temporal evaluation: train on older fights, test on the most recent 15%.
Includes an ablation with/without betting-odds features, plus a
market-only baseline (just follow the bookmaker favorite).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.utils import impute_by_train_median, load_ufc_data, print_class_balance, temporal_split
from core.strategies import MonotonicGBMStrategy, mirror_rows, time_decay_weights

ODDS_COLS = ["a_implied_prob", "diff_implied_prob"]


def make_models():
    return {
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
        "random forest": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=10,
            max_features="sqrt", n_jobs=-1, random_state=42),
        "hist gradient boost": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=42),
    }


def evaluate(name, proba, y_test):
    preds = (proba >= 0.5).astype(int)
    print(f"{name:22s}  acc: {accuracy_score(y_test, preds):.3f}  "
          f"auc: {roc_auc_score(y_test, proba):.3f}  "
          f"logloss: {log_loss(y_test, proba):.3f}")


def run_suite(tag, Xtr_df, Xte_df, y_train, y_test, feature_names):
    X_train, X_test = impute_by_train_median(Xtr_df, Xte_df)
    print(f"\n--- models ({tag}, {len(feature_names)} features) ---")
    fitted = {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        evaluate(name, model.predict_proba(X_test)[:, 1], y_test)
        fitted[name] = model
    return fitted, X_test


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=None,
                    help="drop fights before this year (e.g. 2010)")
    ap.add_argument("--half-life", type=float, default=6.0,
                    help="time-decay half-life in years (0 = off)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="disable mirror augmentation")
    args = ap.parse_args()

    print("loading data...")
    X_df, y, dates = load_ufc_data()
    if args.since:
        keep = (dates >= f"{args.since}-01-01").to_numpy()
        X_df, y, dates = X_df[keep], y[keep], dates[keep]
        print(f"--since {args.since}: {len(y)} fights kept")
    print(f"{X_df.shape[0]} fights, {X_df.shape[1]} features")
    print_class_balance(y, "full")

    Xtr_df, Xte_df, y_train, y_test = temporal_split(X_df, y, dates)

    # naive + market baselines
    print("\n--- baselines ---")
    print(f"{'always class 1':22s}  acc: {y_test.mean():.3f}")
    wr = (Xte_df["diff_win_rate"] > 0).astype(int)
    print(f"{'higher prior win rate':22s}  acc: {accuracy_score(y_test, wr):.3f}")
    market = Xte_df["a_implied_prob"]
    has_odds = market.notna()
    if has_odds.any():
        mkt_acc = accuracy_score(y_test[has_odds], (market[has_odds] > 0.5).astype(int))
        print(f"{'bookmaker favorite':22s}  acc: {mkt_acc:.3f}  "
              f"(on {has_odds.sum()}/{len(y_test)} fights with odds)")

    # mirror augmentation + time-decay weights (train side only)
    dates_tr = dates[Xtr_df.index]
    w_tr = time_decay_weights(dates_tr, args.half_life)
    Xtr_aug, ytr_aug, w_aug = (Xtr_df, y_train, w_tr) if args.no_mirror else \
        mirror_rows(Xtr_df, y_train, w_tr)
    print(f"train rows after augmentation: {len(ytr_aug)} "
          f"(mirror={'off' if args.no_mirror else 'on'}, "
          f"half-life={args.half_life}y)")

    # monotonic GBM: native NaN, domain constraints, augmented + weighted
    mono = MonotonicGBMStrategy(X_df.columns.tolist())
    mono.fit(Xtr_aug, ytr_aug, w_aug)
    evaluate("gbm monotonic", mono.predict_proba(Xte_df), y_test)

    # ablation: stats only vs stats + odds
    no_odds = [c for c in X_df.columns if c not in ODDS_COLS]
    run_suite("stats only", Xtr_df[no_odds], Xte_df[no_odds], y_train, y_test, no_odds)
    fitted, X_test = run_suite("stats + odds", Xtr_df, Xte_df, y_train, y_test,
                               X_df.columns.tolist())

    # method of victory (decision / KO / submission), temporal split
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    df = pd.read_csv("stats/train_dataset.csv", parse_dates=["date"])
    m_train = df.loc[Xtr_df.index, "method"].to_numpy()
    m_test = df.loc[Xte_df.index, "method"].to_numpy()
    mm = HGB(max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)
    mm.fit(Xtr_df.to_numpy(dtype=float), m_train)
    m_pred = mm.predict(Xte_df.to_numpy(dtype=float))
    base = max(pd.Series(m_test).value_counts(normalize=True))
    print(f"\nmethod of victory (dec/ko/sub):  acc {accuracy_score(m_test, m_pred):.3f}"
          f"  vs always-majority {base:.3f}")

    rf = fitted["random forest"]
    print("\ntop 15 features (random forest, stats + odds):")
    names = X_df.columns.tolist()
    order = np.argsort(rf.feature_importances_)[::-1][:15]
    for i in order:
        print(f"  {names[i]:26s} {rf.feature_importances_[i]:.4f}")


if __name__ == "__main__":
    main()
