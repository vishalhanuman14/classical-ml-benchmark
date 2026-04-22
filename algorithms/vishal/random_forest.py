"""
Random Forest using bootstrap aggregation over Decision Trees.
Author: Vishal Hanuman (CS589 HW3)
"""
import numpy as np
from collections import Counter
from algorithms.vishal.decision_tree import build, predict as dt_predict


def _bootstrap(X, y):
    idx = np.random.choice(len(y), size=len(y), replace=True)
    return X[idx], y[idx]


def train(X_train, y_train, feature_types, ntree, max_depth=15, min_size=2):
    m = max(1, int(np.ceil(np.sqrt(X_train.shape[1]))))
    feat_idx = list(range(X_train.shape[1]))
    trees = []
    for _ in range(ntree):
        Xb, yb = _bootstrap(X_train, y_train)
        trees.append(build(Xb, yb, feature_types, feat_idx, max_depth, min_size, random_m=m))
    return trees


def predict(trees, X):
    votes = np.array([dt_predict(t, X) for t in trees])
    return np.array([Counter(votes[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
