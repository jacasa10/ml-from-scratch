from collections import Counter

import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None, root=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    def fit(self, X, y):
        self.num_features = (
            X.shape[1] if not self.num_features else min(self.num_features, X.shape[1])
        )
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # check stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.num_features, replace=False)
        # find best split
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)
        # create child nodes
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_idx in feat_idxs:
            feature = X[:, feature_idx]
            for threshold in np.unique(feature):
                gain = self._information_gain(y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def _information_gain(self, y, feature, threshold):
        parent_entropy = self._entropy(y)
        left, right = self._split(feature, threshold)
        if len(left) == 0 or len(right) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left), len(right)
        children = (n_l / n) * self._entropy(y[left]) + (n_r / n) * self._entropy(y[right])
        return parent_entropy - children

    def _split(self, feature, threshold):
        left = np.argwhere(feature <= threshold).flatten()
        right = np.argwhere(feature > threshold).flatten()
        return left, right

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


if __name__ == "__main__":
    pass
