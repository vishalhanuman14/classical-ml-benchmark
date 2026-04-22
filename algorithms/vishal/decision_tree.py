"""
Decision Tree (ID3 / information gain) supporting numerical and categorical features.
Author: Vishal Hanuman (CS589 HW1 + HW3)
"""
import numpy as np
from collections import Counter


def _entropy(labels):
    total = len(labels)
    return -sum((c / total) * np.log2(c / total)
                for c in Counter(labels).values() if c > 0)


def _majority(labels):
    return Counter(labels).most_common(1)[0][0]


def _gain_categorical(col, y):
    base = _entropy(y)
    weighted = sum(
        (np.sum(col == v) / len(y)) * _entropy(y[col == v])
        for v in np.unique(col)
    )
    return base - weighted


def _gain_numerical(col, y, threshold):
    base = _entropy(y)
    left, right = col <= threshold, col > threshold
    if not left.any() or not right.any():
        return 0.0
    n = len(y)
    return base - (np.sum(left) / n) * _entropy(y[left]) - (np.sum(right) / n) * _entropy(y[right])


def build(X, y, feature_types, feature_indices, max_depth=20, min_size=2, depth=0, random_m=None):
    """
    feature_types: list of 'num' or 'cat' for each column in X.
    random_m: number of candidate features to sample (for Random Forest).
    """
    if len(np.unique(y)) == 1:
        return y[0]
    if not feature_indices or len(y) < min_size or depth >= max_depth:
        return _majority(y)

    candidates = feature_indices
    if random_m is not None and random_m < len(feature_indices):
        candidates = list(np.random.choice(feature_indices, size=random_m, replace=False))

    best_gain, best_feat, best_thresh = -1, None, None
    for f in candidates:
        if feature_types[f] == 'cat':
            g = _gain_categorical(X[:, f], y)
            if g > best_gain:
                best_gain, best_feat, best_thresh = g, f, None
        else:
            col = X[:, f].astype(float)
            thresh = float(np.median(col))
            g = _gain_numerical(col, y, thresh)
            if g > best_gain:
                best_gain, best_feat, best_thresh = g, f, thresh

    if best_gain <= 0:
        return _majority(y)

    node = {
        'feature': best_feat,
        'majority': _majority(y),
        'is_num': feature_types[best_feat] == 'num',
        'children': {},
    }

    if node['is_num']:
        node['threshold'] = best_thresh
        col = X[:, best_feat].astype(float)
        for key, mask in [('<=', col <= best_thresh), ('>', col > best_thresh)]:
            if mask.any():
                node['children'][key] = build(
                    X[mask], y[mask], feature_types, feature_indices,
                    max_depth, min_size, depth + 1, random_m)
    else:
        remaining = [f for f in feature_indices if f != best_feat]
        for v in np.unique(X[:, best_feat]):
            mask = X[:, best_feat] == v
            if mask.any():
                node['children'][v] = build(
                    X[mask], y[mask], feature_types, remaining,
                    max_depth, min_size, depth + 1, random_m)

    return node


def _predict_one(node, x):
    if not isinstance(node, dict):
        return node
    f = node['feature']
    if node['is_num']:
        key = '<=' if float(x[f]) <= node['threshold'] else '>'
    else:
        key = x[f]
    child = node['children'].get(key)
    if child is None:
        return node['majority']
    return _predict_one(child, x)


def predict(tree, X):
    return np.array([_predict_one(tree, x) for x in X])
