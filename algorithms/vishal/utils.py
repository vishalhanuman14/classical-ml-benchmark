"""
Shared utilities: normalization, stratified k-fold, metrics, one-hot encoding.
Author: Vishal Hanuman (CS589 HW1 / HW3)
"""
import numpy as np
from collections import Counter


def normalize(X_train, X_test):
    """Min-max normalization fitted on training data."""
    lo = np.min(X_train, axis=0)
    hi = np.max(X_train, axis=0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (X_train - lo) / rng, (X_test - lo) / rng


def stratified_k_fold(y, k=10):
    """Return list of k fold index arrays, stratified by class."""
    classes = np.unique(y)
    class_idx = {}
    for c in classes:
        idx = np.where(y == c)[0].copy()
        np.random.shuffle(idx)
        class_idx[c] = idx

    folds = [[] for _ in range(k)]
    for c in classes:
        for i, chunk in enumerate(np.array_split(class_idx[c], k)):
            folds[i].extend(chunk.tolist())

    for fold in folds:
        np.random.shuffle(fold)

    return folds


def compute_metrics(y_true, y_pred, pos_label):
    """Return (accuracy, f1) for binary or multiclass (macro F1)."""
    classes = np.unique(y_true)
    acc = np.mean(y_true == y_pred)

    if len(classes) == 2:
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        f1_scores = []
        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        f1 = float(np.mean(f1_scores))

    return acc, f1


def cross_validate(model_fn, X, y, pos_label, k=10):
    """
    Run stratified k-fold CV.
    model_fn(X_train, y_train, X_test) -> y_pred
    Returns mean accuracy and mean F1.
    """
    folds = stratified_k_fold(y, k)
    accs, f1s = [], []
    for i in range(k):
        test_idx  = np.array(folds[i])
        train_idx = np.array([idx for j in range(k) if j != i for idx in folds[j]])
        y_pred = model_fn(X[train_idx], y[train_idx], X[test_idx])
        acc, f1 = compute_metrics(y[test_idx], y_pred, pos_label)
        accs.append(acc)
        f1s.append(f1)
    return float(np.mean(accs)), float(np.mean(f1s))


def one_hot_encode(X_str, cat_indices, num_indices):
    """
    Convert a mixed string array to a float array.
    Categorical columns -> one-hot; numerical columns -> float passthrough.
    Returns encoded float array and a description of the encoding.
    """
    parts = []
    encoding_info = {}

    for f in range(X_str.shape[1]):
        col = X_str[:, f]
        if f in cat_indices:
            vals = sorted(set(col))
            encoding_info[f] = vals
            oh = np.zeros((len(col), len(vals)), dtype=float)
            for j, v in enumerate(vals):
                oh[:, j] = (col == v).astype(float)
            parts.append(oh)
        else:
            parts.append(col.astype(float).reshape(-1, 1))

    return np.hstack(parts)
