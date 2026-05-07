import numpy as np
import math

class GaussianNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.stds = {}
        self.epsilon = epsilon

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        n = len(y_train)
        for c in self.classes:
            mask = y_train == c
            self.priors[c] = np.sum(mask) / n
            self.means[c] = np.mean(X_train[mask], axis=0)
            self.stds[c]  = np.std(X_train[mask], axis=0) + self.epsilon
        return self

    def _log_likelihood(self, x, mean, std):
        return -0.5 * np.log(2 * np.pi * std ** 2) - ((x - mean) ** 2) / (2 * std ** 2)

    def _predict_one(self, x):
        best_class, best_score = None, float('-inf')
        for c in self.classes:
            score = math.log(self.priors[c]) + float(np.sum(self._log_likelihood(x, self.means[c], self.stds[c])))
            if score > best_score:
                best_score, best_class = score, c
        return best_class

    def predict(self, X_test):
        return np.array([self._predict_one(x) for x in X_test])
