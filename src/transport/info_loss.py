"""OT + Information Loss domain adaptation (proposed method)."""

import numpy as np
import ot

from ..losses.information_loss import (
    compute_divergence_matrix,
    information_loss,
    one_hot,
)


def info_loss_domain_adaptation(
    X_s, y_s, X_t, reg_e=1e-1, alpha=1.0, numItermax=50, verbose=False
):
    """
    Domain adaptation using optimal transport with information loss preservation.

    The cost matrix is iteratively refined: source-target pairs whose mapping
    would violate information preservation (same target <- different labels)
    are penalized.

    Parameters
    ----------
    X_s : ndarray, shape (n_source, d)
    y_s : ndarray, shape (n_source,)
    X_t : ndarray, shape (n_target, d)
    reg_e : float
        Sinkhorn entropy regularization.
    alpha : float
        Weight of the information loss term (for logging only; the iterative
        cost-matrix adjustment embodies the optimization).
    numItermax : int
        Number of outer refinement iterations.
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    X_t_adapted : ndarray, shape (n_target, d)
    gamma : ndarray, shape (n_source, n_target)
    """
    n_source, n_target = X_s.shape[0], X_t.shape[0]
    n_classes = len(np.unique(y_s))

    M = ot.dist(X_s, X_t, metric="sqeuclidean")
    M /= M.max()

    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    label_distributions = one_hot(y_s, n_classes)
    H = compute_divergence_matrix(label_distributions)

    gamma = ot.sinkhorn(a, b, M, reg=reg_e, numItermax=1000)

    for it in range(numItermax):
        info_loss_val = information_loss(gamma, H)
        transport_cost = np.sum(gamma * M)
        total = transport_cost + alpha * info_loss_val

        if verbose and it % 5 == 0:
            print(
                f"  Iter {it:3d}  transport={transport_cost:.4f}  "
                f"info_loss={info_loss_val:.4f}  total={total:.4f}"
            )

        if it > 0 and it % 5 == 0:
            pred_labels = np.array(
                [y_s[np.argmax(gamma[:, j])] for j in range(n_target)]
            )
            M_mod = M.copy()
            for c in range(n_classes):
                s_idx = np.where(y_s == c)[0]
                t_idx = np.where(pred_labels == c)[0]
                if len(t_idx) == 0:
                    continue
                M_mod[np.ix_(s_idx, t_idx)] *= 0.9
                for other_c in range(n_classes):
                    if other_c != c:
                        s_other = np.where(y_s == other_c)[0]
                        M_mod[np.ix_(s_other, t_idx)] *= 1.1

            gamma = ot.sinkhorn(a, b, M_mod, reg=reg_e, numItermax=500)

    col_sums = gamma.sum(axis=0, keepdims=True)
    col_sums[col_sums < 1e-8] = 1e-8
    X_t_adapted = (gamma / col_sums).T @ X_s
    X_t_adapted = np.nan_to_num(X_t_adapted)

    return X_t_adapted, gamma
