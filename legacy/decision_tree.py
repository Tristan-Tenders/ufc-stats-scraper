import numpy as np


class TreeNode:
    def __init__(self, feature_idx, feature_val, prediction_probs, information_gain):
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 5,
        min_information_gain: float = 0.0,
        numb_of_features_splitting: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.tree = None

    def _entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p > 0])

    def _find_label_probs(self, data: np.array) -> np.array:
        labels = data[:, -1]
        ar = []
        for c in self.labels_in_train:
            prob = np.sum(labels == c) / len(labels)
            ar.append(prob)
        return np.array(ar)

    def _partition_entropy(self, groups: list) -> float:
        total = sum(len(g) for g in groups)
        entropy = 0.0
        for group in groups:
            if len(group) == 0:
                continue
            probs = [np.sum(group == cls) / len(group) for cls in self.labels_in_train]
            entropy += (len(group) / total) * self._entropy(probs)
        return entropy

    def _split(self, data: np.array, feature_idx: int, feature_val: float):
        mask = data[:, feature_idx] < feature_val
        return data[mask], data[~mask]

    def _find_best_split(self, data: np.array):
        n_features = data.shape[1] - 1
        best_entropy = float("inf")
        best = None

        if self.numb_of_features_splitting is not None:
            feature_indices = np.random.choice(n_features, size=self.numb_of_features_splitting, replace=False)
        else:
            feature_indices = range(n_features)

        for idx in feature_indices:
            sorted_vals = np.unique(data[:, idx])
            candidates = (sorted_vals[:-1] + sorted_vals[1:]) / 2
            for threshold in candidates:
                left, right = self._split(data, idx, threshold)
                if left.shape[0] == 0 or right.shape[0] == 0:
                    continue
                entropy = self._partition_entropy([left[:, -1], right[:, -1]])
                if entropy < best_entropy:
                    best_entropy = entropy
                    best = (left, right, idx, threshold, entropy)

        return best

    def _predict_one_sample(self, X: np.array) -> np.array:
        node = self.tree
        last_probs = None
        while node is not None:
            last_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return last_probs

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        if current_depth >= self.max_depth:
            return None

        result = self._find_best_split(data)
        if result is None:
            return None

        left, right, feat_idx, feat_val, split_entropy = result

        label_probs = self._find_label_probs(data)
        node_entropy = self._entropy(label_probs)
        information_gain = node_entropy - split_entropy

        node = TreeNode(feat_idx, feat_val, label_probs, information_gain)

        if (
            left.shape[0] < self.min_samples_leaf
            or right.shape[0] < self.min_samples_leaf
            or information_gain < self.min_information_gain
        ):
            return node

        node.left = self._create_tree(left, current_depth + 1)
        node.right = self._create_tree(right, current_depth + 1)
        return node

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        self.labels_in_train = np.unique(Y_train)
        data = np.concatenate((X_train, Y_train.reshape(-1, 1)), axis=1)
        self.tree = self._create_tree(data, current_depth=0)

    def predict_proba(self, X_set: np.array) -> np.array:
        return np.apply_along_axis(self._predict_one_sample, 1, X_set)

    def predict(self, X_set: np.array) -> np.array:
        return np.argmax(self.predict_proba(X_set), axis=1)
