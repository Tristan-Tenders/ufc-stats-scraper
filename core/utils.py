import numpy as np
import pandas as pd

META_COLS = ["label", "date", "weightclass", "method", "a_name", "b_name",
             "a_odds_am", "b_odds_am", "sig_strike_diff", "fight_secs"]


def load_ufc_data(path: str = "stats/train_dataset.csv"):
    """Returns X (DataFrame), y, dates — no imputation here (done train-only in split)."""
    df = pd.read_csv(path, parse_dates=["date"])
    y = df["label"].to_numpy(dtype=int)
    dates = df["date"]
    X = df.drop(columns=[c for c in META_COLS if c in df.columns])
    return X, y, dates


def temporal_split(X, y, dates, test_frac: float = 0.15):
    """Train on older fights, test on the most recent ones. No random leakage."""
    cutoff = dates.quantile(1 - test_frac)
    train_mask = (dates < cutoff).to_numpy()
    test_mask = ~train_mask
    print(f"cutoff date: {cutoff.date()}  "
          f"(train: {train_mask.sum()}, test: {test_mask.sum()})")
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def impute_by_train_median(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fill NaNs with medians computed on TRAIN only (no test leakage)."""
    medians = X_train.median(numeric_only=True)
    return (X_train.fillna(medians).fillna(0.0).to_numpy(dtype=float),
            X_test.fillna(medians).fillna(0.0).to_numpy(dtype=float))


def print_class_balance(y, label: str = "") -> None:
    unique, counts = np.unique(y, return_counts=True)
    tag = f"[{label}] " if label else ""
    for cls, cnt in zip(unique, counts):
        print(f"{tag}Class {cls}: {cnt} ({cnt / len(y) * 100:.1f}%)")
