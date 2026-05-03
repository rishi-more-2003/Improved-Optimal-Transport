"""
Information-theoretic loss functions for optimal transport.

Implements the information loss criterion from the paper:
    L = <M, H>_F
where M = gamma @ gamma^T is the source affinity matrix and H[i,j] = KL(y_i || y_j)
is the pairwise label divergence matrix.
"""

import numpy as np
from scipy.special import rel_entr


def kl_divergence(p, q):
    """Compute KL divergence KL(p || q) between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    return np.sum(rel_entr(p, q))


def one_hot(labels, num_classes=None):
    """Convert integer labels to one-hot encoded vectors."""
    labels = np.asarray(labels, dtype=int)
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1
    return np.eye(num_classes)[labels]


def compute_affinity_matrix(gamma):
    """
    Compute the source affinity matrix M = gamma @ gamma^T.

    M[i,j] indicates how strongly source points i and j are mapped to
    the same target points.
    """
    return gamma @ gamma.T


def compute_divergence_matrix(label_distributions, mask=None):
    """
    Compute pairwise KL divergence matrix H.

    Parameters
    ----------
    label_distributions : ndarray, shape (n, n_classes)
        Row i is the label distribution P(y | x_i).
    mask : ndarray, shape (n, n), optional
        Binary mask; H[i,j] is only computed where mask[i,j] == 1.
        Useful for restricting to intra-cluster pairs.

    Returns
    -------
    H : ndarray, shape (n, n)
    """
    n = len(label_distributions)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if mask is not None and mask[i, j] == 0:
                continue
            p = label_distributions[i] + 1e-10
            q = label_distributions[j] + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            H[i, j] = np.sum(p * np.log(p / q))
    return H


def information_loss(gamma, H):
    """
    Compute information loss L = <M, H>_F.

    Parameters
    ----------
    gamma : ndarray, shape (n_source, n_target)
        Transport plan.
    H : ndarray, shape (n_source, n_source)
        Pairwise label divergence matrix.

    Returns
    -------
    loss : float
    """
    M = compute_affinity_matrix(gamma)
    return np.sum(M * H)
