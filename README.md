# Classical ML Benchmark

Benchmarking from-scratch ML algorithm implementations across four datasets as part of COMPSCI 589 (Machine Learning) at UMass Amherst.

**No ML libraries used for training or inference.** The algorithm implementations are from scratch. `numpy` is used for numerical work, `matplotlib` for plots, `pandas` for CSV loading, and `scikit-learn` only for dataset loading.

## Datasets

| # | Dataset | Instances | Features | Task |
|---|---------|-----------|----------|------|
| 1 | Hand-Written Digits | 1,797 | 64 numerical | 10-class classification |
| 2 | Parkinson's Disease | 195 | 22 numerical | Binary (healthy / Parkinson's) |
| 3 | Rice Grains | 3,810 | 7 numerical | Binary (Cammeo / Osmancik) |
| 4 | Credit Approval | 653 | 6 num + 9 cat | Binary (approved / not) |

## Algorithms

| Algorithm | Source | Author |
|-----------|--------|--------|
| k-Nearest Neighbors | HW1 | Vishal |
| Decision Tree (entropy, numerical + categorical) | HW1 + HW3 | Vishal |
| Gaussian Naive Bayes | HW2 (adapted) | Vishal |
| Random Forest | HW3 | Vishal |
| Neural Network | HW4 | Vishal |
| k-Nearest Neighbors, Decision Tree, Gaussian Naive Bayes, Random Forest | HW1-HW3 | Akshith Karthik |

## Project Structure

```
classical-ml-benchmark/
├── algorithms/
│   ├── vishal/         # Vishal's implementations
│   │   ├── knn.py
│   │   ├── decision_tree.py
│   │   ├── naive_bayes.py
│   │   ├── random_forest.py
│   │   └── utils.py    # stratified CV, metrics, normalization
│   └── akshit/         # Akshith's implementations
├── data/               # parkinsons.csv, rice.csv, credit_approval.csv
├── notebooks/          # one Colab-ready notebook per dataset
├── report/             # LaTeX source
└── results/            # generated plots (gitignored PNGs)
```

## Running on Google Colab

Each notebook auto-detects Colab and clones this repo. Just open any notebook via the link below and run the first cell:

| Notebook | Open in Colab |
|----------|---------------|
| 01 – Digits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalhanuman14/classical-ml-benchmark/blob/main/notebooks/01_digits.ipynb) |
| 02 – Parkinson's | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalhanuman14/classical-ml-benchmark/blob/main/notebooks/02_parkinsons.ipynb) |
| 03 – Rice | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalhanuman14/classical-ml-benchmark/blob/main/notebooks/03_rice.ipynb) |
| 04 – Credit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalhanuman14/classical-ml-benchmark/blob/main/notebooks/04_credit.ipynb) |
| **EC#2** – Olivetti Faces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalhanuman14/classical-ml-benchmark/blob/main/notebooks/05_olivetti_ec2.ipynb) |

### Saving work back from Colab

At the end of a Colab session, commit changes with:

```python
import subprocess
subprocess.run(['git', 'config', 'user.email', 'your@email.com'])
subprocess.run(['git', 'config', 'user.name', 'Your Name'])
subprocess.run(['git', 'add', '-A'])
subprocess.run(['git', 'commit', '-m', 'update results'])
# Then push via Colab's built-in GitHub integration or with a PAT
```

## Running Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name classical-ml-benchmark --display-name "Python (.venv: classical-ml-benchmark)"
jupyter notebook
```

For VS Code/Cursor, open the repo root and select `Python (.venv: classical-ml-benchmark)` or `.venv/bin/python` as the notebook kernel. The repo includes `.vscode/settings.json` so notebooks run with the repo root as their working directory.

Vishal-owned notebooks:

- `notebooks/01_digits.ipynb`
- `notebooks/02_parkinsons.ipynb`
- `notebooks/05_olivetti_ec2.ipynb` *(Extra Credit #2 — image classification)*

## Evaluation Protocol

- **Cross-validation:** Stratified 10-fold
- **Metrics:** Accuracy + F1-score
- **Hyperparameters:** ≥6 settings evaluated per algorithm per dataset

## Extra Credit

- **EC#1:** More than two algorithms evaluated on each of the four required datasets.
- **EC#2:** Olivetti Faces image-classification dataset evaluated with k-NN and Neural Network.

## Contributors

- Vishal Hanuman — Datasets 1 & 2
- Akshith Karthik — Datasets 3 & 4
