"""
k-Nearest Neighbors classifier.
Author: Vishal Hanuman (CS589 HW1)
"""
import numpy as np
from collections import Counter


def _distances(X_train, point):
    return np.sqrt(np.sum((X_train - point) ** 2, axis=1))


def predict(X_train, y_train, X_test, k):
    results = []
    for point in X_test:
        dists = _distances(X_train, point)
        k_labels = y_train[np.argsort(dists)[:k]]
        results.append(Counter(k_labels).most_common(1)[0][0])
    return np.array(results)
