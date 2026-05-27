import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_split


def load_ufc_data(path: str = "stats/matchups.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["a_name", "b_name"], errors="ignore")

    Y = df.pop("label").to_numpy(dtype=int)
    X = df.to_numpy(dtype=float)

    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    return X, Y


def split_data(X, Y, test_size: float = 0.2, random_state: int = 42):
    return sklearn_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)


def print_class_balance(Y, label: str = "") -> None:
    unique, counts = np.unique(Y, return_counts=True)
    tag = f"[{label}] " if label else ""
    for cls, cnt in zip(unique, counts):
        print(f"{tag}Class {cls}: {cnt} ({cnt/len(Y)*100:.1f}%)")
