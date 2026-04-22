"""
Gaussian Naive Bayes for numerical features.
Follows the same structure as HW2 Multinomial NB -- priors + per-feature
likelihoods, log-probability prediction.
Author: Vishal Hanuman (CS589 HW2, adapted for continuous features)
"""
import numpy as np
import math


def train(X_train, y_train):
    classes = np.unique(y_train)
    n = len(y_train)
    model = {'classes': classes, 'priors': {}, 'means': {}, 'stds': {}}
    for c in classes:
        mask = y_train == c
        model['priors'][c] = np.sum(mask) / n
        model['means'][c] = np.mean(X_train[mask], axis=0)
        model['stds'][c]  = np.std(X_train[mask], axis=0) + 1e-9
    return model


def _log_likelihood(x, mean, std):
    return -0.5 * np.log(2 * np.pi * std ** 2) - ((x - mean) ** 2) / (2 * std ** 2)


def _predict_one(model, x):
    best_class, best_score = None, float('-inf')
    for c in model['classes']:
        score = math.log(model['priors'][c]) + float(np.sum(_log_likelihood(x, model['means'][c], model['stds'][c])))
        if score > best_score:
            best_score, best_class = score, c
    return best_class


def predict(model, X_test):
    return np.array([_predict_one(model, x) for x in X_test])
