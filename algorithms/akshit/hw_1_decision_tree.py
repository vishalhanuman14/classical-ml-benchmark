import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=10, min_size_for_split=2):
        self.max_depth = max_depth
        self.min_size = min_size_for_split
        self.tree = None
        self.feature_types = None

    def _entropy(self, labels):
        total = len(labels)
        if total == 0: return 0
        return -sum((c / total) * np.log2(c / total)
                    for c in Counter(labels).values() if c > 0)

    def _majority(self, labels):
        return Counter(labels).most_common(1)[0][0]

    def _gain_numerical(self, col, y, threshold):
        base = self._entropy(y)
        left, right = col <= threshold, col > threshold
        if not left.any() or not right.any():
            return 0.0
        n = len(y)
        return base - (np.sum(left) / n) * self._entropy(y[left]) - (np.sum(right) / n) * self._entropy(y[right])

    def _build(self, X, y, feature_indices, depth=0):
        # Base cases: all same class, reached max depth, or too small to split
        if len(np.unique(y)) == 1:
            return y[0]
        if not feature_indices or len(y) < self.min_size or depth >= self.max_depth:
            return self._majority(y)

        best_gain, best_feat, best_thresh = -1, None, None

        # Optimized split search
        for f in feature_indices:
            col = X[:, f].astype(float)
            thresh = float(np.median(col)) # Using median for fast numerical splits
            g = self._gain_numerical(col, y, thresh)
            if g > best_gain:
                best_gain, best_feat, best_thresh = g, f, thresh

        if best_gain <= 0:
            return self._majority(y)

        # Build the current node
        node = {
            'feature': best_feat,
            'threshold': best_thresh,
            'majority': self._majority(y),
            'children': {}
        }

        # Binary split for numerical features
        col = X[:, best_feat].astype(float)
        for key, mask in [('<=', col <= best_thresh), ('>', col > best_thresh)]:
            if mask.any():
                node['children'][key] = self._build(X[mask], y[mask], feature_indices, depth + 1)
            else:
                node['children'][key] = node['majority']

        return node

    def fit(self, X, y):
        # Convert to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
        feature_indices = list(range(X.shape[1]))
        self.tree = self._build(X, y, feature_indices)
        return self

    def _predict_one(self, node, x):
        if not isinstance(node, dict):
            return node

        f = node['feature']
        key = '<=' if float(x[f]) <= node['threshold'] else '>'
        child = node['children'].get(key)

        if child is None:
            return node['majority']
        return self._predict_one(child, x)

    def predict(self, X):
        return np.array([self._predict_one(self.tree, x) for x in X])
