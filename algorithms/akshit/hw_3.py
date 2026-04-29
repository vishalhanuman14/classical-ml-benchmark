import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter
 
 
def evaluate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
 
    # CHANGED: guard against empty arrays
    if len(y_true) == 0:
        return 0, 0, 0, 0
 
    classes = np.unique(y_true)
 
    # CHANGED: guard against empty classes
    if len(classes) == 0:
        return 0, 0, 0, 0
 
    pos_class = classes[1] if len(classes) > 1 else classes[0]
 
    tp = np.sum((y_pred == pos_class) & (y_true == pos_class))
    fp = np.sum((y_pred == pos_class) & (y_true != pos_class))
    fn = np.sum((y_pred != pos_class) & (y_true == pos_class))
    tn = np.sum((y_pred != pos_class) & (y_true != pos_class))
 
    accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
 
    return accuracy, precision, recall, f1
 
 
def stratified_kfold(X, y, k=5):
    y_array = np.array(y)
    classes = np.unique(y_array)
 
    # CHANGED: list() so np.random.shuffle works in-place correctly
    class_indices = [list(np.where(y_array == c)[0]) for c in classes]
 
    for indices in class_indices:
        np.random.shuffle(indices)
 
    folds = [[] for _ in range(k)]
    for indices in class_indices:
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)
 
    splits = []
    for i in range(k):
        test_idx  = folds[i]
        train_idx = []
        for j in range(k):
            if i != j:
                train_idx.extend(folds[j])
        # CHANGED: dtype=int so numpy can use these as array indices
        splits.append((np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)))
 
    return splits
 
 
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 *, value=None, is_categorical=False):
        self.feature        = feature
        self.threshold      = threshold
        self.left           = left
        self.right          = right
        self.value          = value
        self.is_categorical = is_categorical
 
    def is_leaf_node(self):
        return self.value is not None
 
 
class DecisionTree:
    # CHANGED: default minimum_gain=0.0 so trees actually learn
    def __init__(self, min_size_for_split=2, max_depth=100, minimum_gain=0.0, m_features=None):
        self.min_size_for_split = min_size_for_split
        self.max_depth          = max_depth
        self.minimum_gain       = minimum_gain
        self.m_features         = m_features
        self.root               = None
 
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.m_features = self.m_features if self.m_features else self.n_features
        self.root = self._grow_tree(X, y)
 
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
 
        # CHANGED: added n_samples == 0 check
        if (n_samples == 0 or depth >= self.max_depth or
                n_labels == 1 or n_samples < self.min_size_for_split):
            return Node(value=self._most_common_label(y))
 
        feat_idxs = np.random.choice(n_feats, self.m_features, replace=False)
 
        best_feature, best_thresh, is_cat = self._best_split(X, y, feat_idxs)
 
        if best_feature is None:
            return Node(value=self._most_common_label(y))
 
        if is_cat:
            left_idxs  = np.where(X[:, best_feature] == best_thresh)[0]
            right_idxs = np.where(X[:, best_feature] != best_thresh)[0]
        else:
            left_idxs  = np.where(X[:, best_feature] <= best_thresh)[0]
            right_idxs = np.where(X[:, best_feature] >  best_thresh)[0]
 
        # CHANGED: guard against empty splits
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=self._most_common_label(y))
 
        left  = self._grow_tree(X[left_idxs],  y[left_idxs],  depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right, is_categorical=is_cat)
 
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold, split_is_cat = None, None, False
 
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
 
            # CHANGED: robust categorical detection via try/except instead of isinstance
            # isinstance fails for numpy.str_ types which pandas produces
            try:
                X_column.astype(float)
                is_categorical = False
            except (ValueError, TypeError):
                is_categorical = True
 
            thresholds = np.unique(X_column)
 
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr, is_categorical)
 
                if gain > best_gain and gain >= self.minimum_gain:
                    best_gain       = gain
                    split_idx       = feat_idx
                    split_threshold = thr
                    split_is_cat    = is_categorical
 
        return split_idx, split_threshold, split_is_cat
 
    def _information_gain(self, y, X_column, threshold, is_categorical):
        parent_entropy = self._entropy(y)
 
        if is_categorical:
            left_idxs  = np.where(X_column == threshold)[0]
            right_idxs = np.where(X_column != threshold)[0]
        else:
            left_idxs  = np.where(X_column <= threshold)[0]
            right_idxs = np.where(X_column >  threshold)[0]
 
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
 
        n        = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l      = self._entropy(y[left_idxs])
        e_r      = self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
 
        return parent_entropy - child_entropy
 
    def _entropy(self, y):
        # CHANGED: guard against empty y
        if len(y) == 0:
            return 0
        counts        = np.bincount(np.unique(y, return_inverse=True)[1])
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
 
    def _most_common_label(self, y):
        # CHANGED: guard against empty y
        if len(y) == 0:
            return None
        counter = Counter(y)
        return counter.most_common(1)[0][0]
 
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
 
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
 
        if node.is_categorical:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:
            # CHANGED: guard against None threshold
            if node.threshold is None:
                return self._traverse_tree(x, node.left)
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
 
 
class RandomForest:
    # CHANGED: default minimum_gain=0.0, max_depth=8, min_size_for_split=5
    def __init__(self, ntree=10, max_depth=8, min_size_for_split=5, minimum_gain=0.0):
        self.ntree              = ntree
        self.max_depth          = max_depth
        self.min_size_for_split = min_size_for_split
        self.minimum_gain       = minimum_gain
        self.trees              = []
 
    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        m_features = int(np.round(np.sqrt(n_features)))
 
        for _ in range(self.ntree):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_size_for_split=self.min_size_for_split,
                minimum_gain=self.minimum_gain,
                m_features=m_features
            )
            indices     = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
 
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
 
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])
 
 
def evaluate_random_forest(X, y, dataset_name):
    ntree_values = [1, 5, 10, 20, 30, 40, 50]
 
    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
 
    folds = stratified_kfold(X, y, k=5)
 
    for ntree in ntree_values:
        print(f"Training Random Forest with ntree={ntree} on {dataset_name}...")
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
 
        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
 
            # CHANGED: minimum_gain=0.0 so trees split and learn
            rf = RandomForest(ntree=ntree, max_depth=8, min_size_for_split=5, minimum_gain=0.0)
            rf.fit(X_train, y_train)
 
            y_pred = rf.predict(X_test)
 
            acc, prec, rec, f1 = evaluate_metrics(y_test, y_pred)
            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1'].append(f1)
 
        results['accuracy'].append(np.mean(fold_metrics['accuracy']))
        results['precision'].append(np.mean(fold_metrics['precision']))
        results['recall'].append(np.mean(fold_metrics['recall']))
        results['f1'].append(np.mean(fold_metrics['f1']))
 
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Performance Metrics vs. Number of Trees ({dataset_name})", fontsize=16)
 
    for idx, ax in enumerate(axes.flatten()):
        metric_key = metrics[idx]
        ax.plot(ntree_values, results[metric_key], marker='o', linestyle='-', color='b')
        ax.set_title(f"{titles[idx]} vs ntree")
        ax.set_xlabel("Number of Trees (ntree)")
        ax.set_ylabel(titles[idx])
        ax.grid(True)
        txt = f"Graph showing the {titles[idx].lower()} of the RF as a function of ntree for {dataset_name}."
        ax.text(0.5, -0.15, txt, transform=ax.transAxes, ha='center', fontsize=9)
 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    # CHANGED: auto-save plots to figures/ folder
    plt.savefig(f"figures/{dataset_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()
 
    return results