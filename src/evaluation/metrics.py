"""Evaluation utilities for domain adaptation experiments."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def evaluate_adaptation(X_s, y_s, X_t, y_t, k=5):
    """
    Train a k-NN classifier on source and evaluate on target.

    This is the standard protocol for measuring domain alignment quality:
    a 1-NN (or k-NN) classifier trained on the (possibly transported)
    source data is tested on the target domain.

    Parameters
    ----------
    X_s : ndarray, shape (n_source, d)
        Source features (raw or transported).
    y_s : ndarray, shape (n_source,)
        Source labels.
    X_t : ndarray, shape (n_target, d)
        Target features.
    y_t : ndarray, shape (n_target,)
        Target labels (ground truth for evaluation).
    k : int
        Number of neighbors.

    Returns
    -------
    accuracy : float
    """
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_s, y_s)
    y_pred = clf.predict(X_t)
    return accuracy_score(y_t, y_pred)
