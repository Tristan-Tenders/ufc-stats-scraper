import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_base_learner: int = 100,
        numb_of_features_splitting: int = 7,
        bootstrap_sample_size: int | None = None,
        max_depth: int = 6,
        min_samples_leaf: int = 5,
        min_information_gain: float = 0.0,
    ):
        self.n_base_learner = n_base_learner
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.base_learner_list = []

    def _create_bootstrap_samples(self, X: np.array, Y: np.array) -> tuple:
        if self.bootstrap_sample_size is None:
            self.bootstrap_sample_size = X.shape[0]
        X_samples, Y_samples = [], []
        for _ in range(self.n_base_learner):
            idx = np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            X_samples.append(X[idx])
            Y_samples.append(Y[idx])
        return X_samples, Y_samples

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        X_samples, Y_samples = self._create_bootstrap_samples(X_train, Y_train)
        for X_sample, Y_sample in zip(X_samples, Y_samples):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_information_gain=self.min_information_gain,
                numb_of_features_splitting=self.numb_of_features_splitting,
            )
            tree.train(X_sample, Y_sample)
            self.base_learner_list.append(tree)

    def predict_proba(self, X_set: np.array) -> np.array:
        return np.mean([tree.predict_proba(X_set) for tree in self.base_learner_list], axis=0)

    def predict(self, X_set: np.array) -> np.array:
        return np.argmax(self.predict_proba(X_set), axis=1)
