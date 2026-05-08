# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math


def shuffle(X, y, random_state=None):
    """Shuffle X and y together without relying on sklearn utilities."""
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    return np.asarray(X)[idx], np.asarray(y)[idx]


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Small local replacement for the HW1 experiment driver."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_test = int(np.ceil(len(X) * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# %%
def normalize(X_train, X_test):
    """Min-max normalize using training statistics."""
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1  # avoid division by zero
    return (X_train - mins) / rng, (X_test - mins) / rng


def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))



# %%
#Question 1: k-NN
def knn_predict(X_train, y_train, X_query, k):
    """Predict labels for X_query using k-NN over X_train/y_train."""
    predictions = []
    for xq in X_query:
        dists = np.sqrt(np.sum((X_train - xq) ** 2, axis=1))
        nn_idx = np.argsort(dists)[:k]
        nn_labels = [y_train[i] for i in nn_idx]
        predictions.append(Counter(nn_labels).most_common(1)[0][0])
    return predictions


def run_knn_experiments(X, y, k_values, n_runs=20, normalize_features=True):
    """Return (train_accs, test_accs) each shape (len(k_values), n_runs)."""
    train_accs = np.zeros((len(k_values), n_runs))
    test_accs  = np.zeros((len(k_values), n_runs))

    for run in range(n_runs):
        X_sh, y_sh = shuffle(X, y, random_state=run)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sh, y_sh, test_size=0.2, random_state=run)

        if normalize_features:
            X_tr_n, X_te_n = normalize(X_tr, X_te)
        else:
            X_tr_n, X_te_n = X_tr, X_te

        for ki, k in enumerate(k_values):
            y_pred_tr = knn_predict(X_tr_n, list(y_tr), X_tr_n, k)
            y_pred_te = knn_predict(X_tr_n, list(y_tr), X_te_n, k)
            train_accs[ki, run] = accuracy(y_tr, y_pred_tr)
            test_accs[ki, run]  = accuracy(y_te, y_pred_te)

        print(f"  k-NN run {run+1}/{n_runs} done", flush=True)

    return train_accs, test_accs


def plot_knn(k_values, accs, title, ylabel, filename):
    means = accs.mean(axis=1)
    stds  = accs.std(axis=1)
    plt.figure(figsize=(8, 5))
    plt.errorbar(k_values, means, yerr=stds, fmt='o-', capsize=4,
                 color='steelblue', ecolor='gray')
    plt.xlabel("Value of k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(k_values[::3])
    plt.ylim(0.5, 1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")

# %%
def load_wdbc(path='/Users/akshithkarthik/Downloads/UMass/CS 589/HW1_CMPSCI_589_Spring2026_Supporting_Files/datasets/wdbc.csv'):
    data = np.genfromtxt(path, delimiter=',')
    X = data[:, :30].astype(float)
    y = data[:, 30].astype(int)
    return X, list(y)

# %%
if __name__ == '__main__':
    k_values = list(range(1, 52, 2))   # 1, 3, ..., 51

    # ── Question 1: k-NN on WDBC ──────────────────────────────
    print("Loading WDBC dataset...")
    X_wdbc, y_wdbc = load_wdbc('/Users/akshithkarthik/Downloads/UMass/CS 589/HW1_CMPSCI_589_Spring2026_Supporting_Files/datasets/wdbc.csv')

    print("Running k-NN experiments (with normalization)...")
    train_accs, test_accs = run_knn_experiments(
        X_wdbc, y_wdbc, k_values, n_runs=20, normalize_features=True)

    plot_knn(k_values, train_accs,
             "k-NN Accuracy on Training Set (with normalization)",
             "Accuracy over training data", "fig_knn_train.png")

    plot_knn(k_values, test_accs,
             "k-NN Accuracy on Testing Set (with normalization)",
             "Accuracy over testing data", "fig_knn_test.png")

    # Q1.6 – no normalization
    print("Running k-NN experiments (WITHOUT normalization)...")
    _, test_accs_nonorm = run_knn_experiments(
        X_wdbc, y_wdbc, k_values, n_runs=20, normalize_features=False)

    plot_knn(k_values, test_accs_nonorm,
             "k-NN Accuracy on Testing Set (NO normalization)",
             "Accuracy over testing data", "fig_knn_test_nonorm.png")

    # Print best k for Q1.5 and Q1.6
    best_k_norm   = k_values[np.argmax(test_accs.mean(axis=1))]
    best_k_nonorm = k_values[np.argmax(test_accs_nonorm.mean(axis=1))]
    print(f"\nBest k (normalized):     {best_k_norm}")
    print(f"Best k (not normalized): {best_k_nonorm}")

# %%
#Question 2: Decision Tree

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return 1 - sum((c / n) ** 2 for c in counts.values())


def information_gain(parent_labels, splits, criterion='entropy'):
    measure = entropy if criterion == 'entropy' else gini
    parent_score = measure(parent_labels)
    n = len(parent_labels)
    weighted = sum(len(s) / n * measure(s) for s in splits)
    return parent_score - weighted


def best_split(X_cols, y, feature_indices, criterion='entropy'):
    """Return (best_feature_idx, best_ig)."""
    best_feat, best_ig = None, -1
    for fi in feature_indices:
        col = [row[fi] for row in X_cols]
        values = set(col)
        splits = [[y[i] for i, v in enumerate(col) if v == val] for val in values]
        ig = information_gain(y, splits, criterion)
        if ig > best_ig:
            best_ig, best_feat = ig, fi
    return best_feat, best_ig


# %%
class DTNode:
    def __init__(self, label=None, feature=None, children=None):
        self.label    = label      # not None => leaf
        self.feature  = feature    # feature index to split on
        self.children = children or {}  # value -> DTNode



def build_tree(X, y, feature_indices, criterion='entropy', prune_threshold=None):
    # All same class
    if len(set(y)) == 1:
        return DTNode(label=y[0])
    # No features left
    if not feature_indices:
        return DTNode(label=Counter(y).most_common(1)[0][0])

    # Pruning heuristic (QE.2)
    if prune_threshold is not None:
        majority_count = Counter(y).most_common(1)[0][1]
        if majority_count / len(y) >= prune_threshold:
            return DTNode(label=Counter(y).most_common(1)[0][0])

    feat, ig = best_split(X, y, feature_indices, criterion)
    if feat is None or ig <= 0:
        return DTNode(label=Counter(y).most_common(1)[0][0])

    col = [row[feat] for row in X]
    values = set(col)
    children = {}
    remaining = [f for f in feature_indices if f != feat]

    for val in values:
        subset_X = [X[i] for i, v in enumerate(col) if v == val]
        subset_y = [y[i] for i, v in enumerate(col) if v == val]
        if not subset_y:
            children[val] = DTNode(label=Counter(y).most_common(1)[0][0])
        else:
            children[val] = build_tree(subset_X, subset_y, remaining,
                                       criterion, prune_threshold)

    return DTNode(feature=feat, children=children,
                  label=Counter(y).most_common(1)[0][0])  # fallback label




# %%
def dt_predict_one(node, x, default_label):
    if node.label is not None and node.feature is None:
        return node.label
    val = x[node.feature]
    if val in node.children:
        child = node.children[val]
        if child.feature is None:
            return child.label
        return dt_predict_one(child, x, default_label)
    return node.label  # fallback


def dt_predict(root, X):
    default = root.label
    return [dt_predict_one(root, x, default) for x in X]


def run_dt_experiments(X, y, n_runs=100, criterion='entropy', prune_threshold=None):
    train_accs, test_accs = [], []
    feat_indices = list(range(len(X[0])))

    for run in range(n_runs):
        X_sh, y_sh = shuffle(X, y, random_state=run)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sh, y_sh, test_size=0.2, random_state=run)

        tree = build_tree(X_tr, y_tr, feat_indices, criterion, prune_threshold)

        y_pred_tr = dt_predict(tree, X_tr)
        y_pred_te = dt_predict(tree, X_te)
        train_accs.append(accuracy(y_tr, y_pred_tr))
        test_accs.append(accuracy(y_te, y_pred_te))

        if (run + 1) % 10 == 0:
            print(f"  DT run {run+1}/{n_runs} done", flush=True)

    return np.array(train_accs), np.array(test_accs)


def plot_histogram(accs, title, xlabel, filename, bins=20):
    mean_acc = accs.mean()
    std_acc  = accs.std()
    plt.figure(figsize=(8, 5))
    plt.hist(accs, bins=bins, color='steelblue', edgecolor='black')
    plt.axvline(x=mean_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {mean_acc:.4f}')
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(f"{title}\nMean={mean_acc:.4f}, Std={std_acc:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename} | mean={mean_acc:.4f} std={std_acc:.4f}")
    return mean_acc, std_acc

# %%
def load_car(path='/Users/akshithkarthik/Downloads/UMass/CS 589/HW1_CMPSCI_589_Spring2026_Supporting_Files/datasets/car.csv'):
    import csv
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    X = [row[:-1] for row in rows]
    y = [row[-1]  for row in rows]
    return X, y

# %%
if __name__ == '__main__':
    k_values = list(range(1, 52, 2))   # 1, 3, ..., 51

    print("\nLoading Car dataset...")
    X_car, y_car = load_car('/Users/akshithkarthik/Downloads/UMass/CS 589/HW1_CMPSCI_589_Spring2026_Supporting_Files/datasets/car.csv')

    print("Running Decision Tree experiments (Information Gain, 100 runs)...")
    dt_train, dt_test = run_dt_experiments(
        X_car, y_car, n_runs=100, criterion='entropy')

    m1, s1 = plot_histogram(dt_train,
        "Decision Tree – Accuracy on Training Set (Info Gain)",
        "Accuracy", "fig_dt_train.png")
    m2, s2 = plot_histogram(dt_test,
        "Decision Tree – Accuracy on Testing Set (Info Gain)",
        "Accuracy", "fig_dt_test.png")
    print(f"DT Train: mean={m1:.4f} std={s1:.4f}")
    print(f"DT Test:  mean={m2:.4f} std={s2:.4f}")

    # ── Extra Credit QE.1: Gini ───────────────────────────────
    print("\nRunning Decision Tree experiments (Gini, 100 runs)...")
    dt_gini_train, dt_gini_test = run_dt_experiments(
        X_car, y_car, n_runs=100, criterion='gini')

    plot_histogram(dt_gini_train,
        "Decision Tree – Accuracy on Training Set (Gini)",
        "Accuracy", "fig_dt_gini_train.png")
    plot_histogram(dt_gini_test,
        "Decision Tree – Accuracy on Testing Set (Gini)",
        "Accuracy", "fig_dt_gini_test.png")

    # ── Extra Credit QE.2: Pruning heuristic (85%) ───────────
    print("\nRunning Decision Tree experiments (85% pruning, 100 runs)...")
    dt_pruned_train, dt_pruned_test = run_dt_experiments(
        X_car, y_car, n_runs=100, criterion='entropy', prune_threshold=0.85)

    plot_histogram(dt_pruned_train,
        "Decision Tree – Accuracy on Training Set (85% Pruning)",
        "Accuracy", "fig_dt_pruned_train.png")
    plot_histogram(dt_pruned_test,
        "Decision Tree – Accuracy on Testing Set (85% Pruning)",
        "Accuracy", "fig_dt_pruned_test.png")

    print("\nAll experiments complete.")


