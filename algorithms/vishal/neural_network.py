"""
Neural Network with sigmoid activations and mini-batch gradient descent.
Supports multi-class via multiple output neurons with one-hot labels.
Author: Vishal Hanuman (CS589 HW4)
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def _sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)


def _add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def init_weights(layer_sizes):
    """Random uniform init in [-0.12, 0.12]."""
    theta = []
    for i in range(len(layer_sizes) - 1):
        theta.append(np.random.uniform(-0.12, 0.12,
                                       (layer_sizes[i + 1], layer_sizes[i] + 1)))
    return theta


def forward(X, theta):
    a_vals, z_vals = [X], []
    cur = X
    for t in theta:
        z = _add_bias(cur) @ t.T
        cur = sigmoid(z)
        z_vals.append(z)
        a_vals.append(cur)
    return a_vals, z_vals


def cost(X, Y, theta, lam=0.0):
    m = X.shape[0]
    a_vals, z_vals = forward(X, theta)
    h = np.clip(a_vals[-1], 1e-12, 1 - 1e-12)
    J = -np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h)) / m
    J += (lam / (2 * m)) * sum(np.sum(t[:, 1:] ** 2) for t in theta)
    return J, a_vals, z_vals


def _backprop_step(X, Y, theta, lam=0.0):
    m = X.shape[0]
    J, a_vals, z_vals = cost(X, Y, theta, lam)

    deltas = [None] * len(theta)
    deltas[-1] = a_vals[-1] - Y
    for i in range(len(theta) - 2, -1, -1):
        deltas[i] = (deltas[i + 1] @ theta[i + 1][:, 1:]) * _sigmoid_deriv(z_vals[i])

    grads = []
    for i in range(len(theta)):
        g = (deltas[i].T @ _add_bias(a_vals[i])) / m
        g[:, 1:] += (lam / m) * theta[i][:, 1:]
        grads.append(g)

    return J, grads


def train(X, Y, layer_sizes, lam=0.0, alpha=0.3, epochs=500, batch_size=32):
    """
    Train network. Y must be one-hot for multi-class (use one_hot() helper).
    Returns (theta, cost_history) where cost_history is list of (epoch, J).
    """
    theta = init_weights(layer_sizes)
    m = X.shape[0]
    history = []

    for ep in range(epochs):
        idx = np.random.permutation(m)
        for start in range(0, m, batch_size):
            batch = idx[start:start + batch_size]
            _, grads = _backprop_step(X[batch], Y[batch], theta, lam)
            for i in range(len(theta)):
                theta[i] -= alpha * grads[i]

        if ep == 0 or (ep + 1) % 50 == 0 or ep == epochs - 1:
            J, _, _ = cost(X, Y, theta, lam)
            history.append((ep + 1, J))

    return theta, history


def predict(theta, X):
    """Argmax over output neurons for multi-class; threshold 0.5 for binary."""
    a_vals, _ = forward(X, theta)
    out = a_vals[-1]
    if out.shape[1] == 1:
        return (out >= 0.5).astype(int).reshape(-1)
    return np.argmax(out, axis=1)


def one_hot(y, n_classes):
    """Convert integer label array to one-hot matrix (n x n_classes)."""
    Y = np.zeros((len(y), n_classes))
    for i, c in enumerate(y):
        Y[i, int(c)] = 1.0
    return Y
