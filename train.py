import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import load_ufc_data, split_data, print_class_balance


def evaluate(preds: np.array, Y_test: np.array) -> None:
    accuracy = (preds == Y_test).mean()
    print(f"\ntest accuracy: {accuracy:.3f}")

    for cls in np.unique(Y_test):
        mask = Y_test == cls
        cls_acc = (preds[mask] == Y_test[mask]).mean()
        print(f"  class {cls}: {cls_acc:.3f}  ({mask.sum()} samples)")


def main():
    print("loading data...")
    X, Y = load_ufc_data("stats/matchups.csv")
    print(f"{X.shape[0]} rows, {X.shape[1]} features")
    print_class_balance(Y, "full")

    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    print(f"\ntrain: {len(X_train)}  test: {len(X_test)}")
    print_class_balance(Y_train, "train")

    print("\ntraining...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features=7,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
    )
    rf.fit(X_train, Y_train)
    print("done.")

    preds = rf.predict(X_test)
    evaluate(preds, Y_test)


if __name__ == "__main__":
    main()
