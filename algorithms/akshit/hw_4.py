# %% [markdown]
# # Dataset 4 – Credit Approval
# **Implementation:** Akshith Karthik (algorithms/akshit/)
# **Algorithms:** Decision Tree and Random Forest
# **Note:** Mixed features — use `utils.one_hot_encode` for categorical columns  
# **Evaluation:** Stratified 10-fold CV · Accuracy · Macro F1

# %%
import sys, os, subprocess

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run(['git', 'clone', f'https://github.com/vishalhanuman14/classical-ml-benchmark.git'], check=True)
    os.chdir('classical-ml-benchmark')

if '.' not in sys.path:
    sys.path.insert(0, '.')


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, '.')

from algorithms.vishal import utils

df = pd.read_csv('data/credit_approval.csv')
print(df.head(2))
print(f"Shape: {df.shape}")
print(f"Dtypes:\n{df.dtypes}")


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom functions from hw_3.py
from algorithms.akshit.hw_3 import DecisionTree, RandomForest, evaluate_metrics, stratified_kfold

# Import the shared utilities
from algorithms.vishal import utils

# --- Data Loading ---
df = pd.read_csv('data/credit_approval.csv')

# Dynamically identify target and features (assuming target is the last column)
target_col = df.columns[-1]
feature_cols = df.columns.drop(target_col)

# Identify categorical and numerical feature indices
cat_columns = [i for i, col in enumerate(feature_cols) if df[col].dtype == object]
num_columns = [i for i, col in enumerate(feature_cols) if df[col].dtype != object]

X_str = df[feature_cols].values.astype(str)
y = df[target_col].values.astype(str)

# Apply one-hot encoding to handle the mixed features
X_enc = utils.one_hot_encode(X_str, cat_indices=cat_columns, num_indices=num_columns)
X_raw = X_enc.astype(float) # Cast to float for mathematical operations

print(f"Instances: {len(y)}, Features (after encoding): {X_raw.shape[1]}")
print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

# %% [markdown]
# ## Algorithm 1: Decision Tree Hyperparameter Sweep

# %%
# --- Algorithm 1 Sweep: Decision Tree ---
depth_values = [2, 4, 6, 8, 10, 12]
dt_results = {'max_depth': [], 'accuracy': [], 'f1': []}

print("Running 10-Fold CV for Decision Tree...")
folds = stratified_kfold(X_raw, y, k=10)

for depth in depth_values:
    acc_list, f1_list = [], []
    for train_idx, test_idx in folds:
        X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Initialize and train Decision Tree
        dt = DecisionTree(max_depth=depth, min_size_for_split=2)
        dt.fit(X_tr, y_tr)

        # Predict and evaluate
        y_pred = dt.predict(X_te)
        acc, prec, rec, f1 = evaluate_metrics(y_te, y_pred)

        acc_list.append(acc)
        f1_list.append(f1)

    dt_results['max_depth'].append(depth)
    dt_results['accuracy'].append(np.mean(acc_list))
    dt_results['f1'].append(np.mean(f1_list))
    print(f"max_depth={depth:<2} | Acc: {np.mean(acc_list):.4f} | F1: {np.mean(f1_list):.4f}")

# %% [markdown]
# ## Algorithm 2: Random Forest Hyperparameter Sweep

# %%
# --- Algorithm 2 Sweep: Random Forest ---
ntree_values = [1, 5, 10, 15, 20, 25]
rf_results = {'ntree': [], 'accuracy': [], 'f1': []}

print("Running 10-Fold CV for Random Forest...")

for ntree in ntree_values:
    acc_list, f1_list = [], []
    for train_idx, test_idx in folds: # Reusing the exact same folds for a fair comparison
        X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Initialize and train Random Forest
        rf = RandomForest(ntree=ntree, max_depth=8)
        rf.fit(X_tr, y_tr)

        # Predict and evaluate
        y_pred = rf.predict(X_te)
        acc, prec, rec, f1 = evaluate_metrics(y_te, y_pred)

        acc_list.append(acc)
        f1_list.append(f1)

    rf_results['ntree'].append(ntree)
    rf_results['accuracy'].append(np.mean(acc_list))
    rf_results['f1'].append(np.mean(f1_list))
    print(f"ntree={ntree:<2} | Acc: {np.mean(acc_list):.4f} | F1: {np.mean(f1_list):.4f}")

# %% [markdown]
# ## Plots and Summary Table

# %%
# --- Plotting Learning Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision Tree Plot
axes[0].plot(dt_results['max_depth'], dt_results['accuracy'], marker='o', label='Accuracy')
axes[0].plot(dt_results['max_depth'], dt_results['f1'], marker='s', label='Macro F1')
axes[0].set_title('Algorithm 1: Decision Tree vs. Max Depth')
axes[0].set_xlabel('Max Depth')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].grid(True)

# Random Forest Plot
axes[1].plot(rf_results['ntree'], rf_results['accuracy'], marker='o', label='Accuracy')
axes[1].plot(rf_results['ntree'], rf_results['f1'], marker='s', label='Macro F1')
axes[1].set_title('Algorithm 2: Random Forest vs. ntree')
axes[1].set_xlabel('Number of Trees (ntree)')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('results/credit_dt_rf_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Summary Table ---
print("\n=== SUMMARY TABLE ===")
print("Decision Tree Results:")
df_dt = pd.DataFrame(dt_results)
print(df_dt.to_string(index=False))

print("\nRandom Forest Results:")
df_rf = pd.DataFrame(rf_results)
print(df_rf.to_string(index=False))
