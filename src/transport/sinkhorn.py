"""Standard Sinkhorn OT domain adaptation (baseline)."""

import numpy as np
import ot


def sinkhorn_domain_adaptation(X_s, y_s, X_t, reg_e=1e-1):
    """
    Domain adaptation via entropic-regularized Sinkhorn optimal transport.

    Parameters
    ----------
    X_s : ndarray, shape (n_source, d)
        Source domain features (standardized).
    y_s : ndarray, shape (n_source,)
        Source domain labels.
    X_t : ndarray, shape (n_target, d)
        Target domain features (standardized).
    reg_e : float
        Entropy regularization strength.

    Returns
    -------
    X_t_adapted : ndarray, shape (n_target, d)
        Adapted target features.
    gamma : ndarray, shape (n_source, n_target)
        Transport plan.
    """
    n_source = X_s.shape[0]
    n_target = X_t.shape[0]

    M = ot.dist(X_s, X_t, metric="sqeuclidean")
    M /= M.max() + 1e-9

    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    gamma = ot.sinkhorn(a, b, M, reg=reg_e)

    col_sums = gamma.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1e-8
    X_t_adapted = (gamma / col_sums).T @ X_s
    X_t_adapted = np.nan_to_num(X_t_adapted)

    return X_t_adapted, gamma
