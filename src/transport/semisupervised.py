"""Semi-supervised OT domain adaptation variants."""

import numpy as np
import ot

from ..losses.information_loss import (
    compute_divergence_matrix,
    information_loss,
    one_hot,
)


def semisupervised_sinkhorn(X_s, y_s, X_t, y_t_partial, reg_e=0.01, reg_cl=0.1):
    """
    Semi-supervised domain adaptation using POT's SinkhornLpl1Transport.

    Parameters
    ----------
    X_s, y_s : source features and labels.
    X_t : target features.
    y_t_partial : target labels; -1 for unlabeled samples.
    reg_e, reg_cl : regularization strengths.

    Returns
    -------
    X_t_adapted : ndarray
    transport : fitted POT transport object
    """
    transp = ot.da.SinkhornLpl1Transport(reg_e=reg_e, reg_cl=reg_cl)
    transp.fit(Xs=X_s, ys=y_s, Xt=X_t, yt=y_t_partial)
    X_t_adapted = transp.transform(X_t)
    return X_t_adapted, transp


def semisupervised_info_loss(
    X_s, y_s, X_t, y_t_partial, reg_e=1e-1, reg_cl=0.1,
    alpha=1.0, numItermax=50, verbose=False,
):
    """
    Semi-supervised OT + information loss.

    Combines partially-labeled target guidance with the information-
    preservation penalty.

    Parameters
    ----------
    X_s, y_s : source features and labels.
    X_t : target features.
    y_t_partial : target labels; -1 for unlabeled.
    reg_e : Sinkhorn regularization.
    reg_cl : class-label cost adjustment strength.
    alpha : information loss weight.
    numItermax : outer iterations.
    verbose : print diagnostics.

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

        if verbose and it % 5 == 0:
            total = transport_cost + alpha * info_loss_val
            print(
                f"  Iter {it:3d}  transport={transport_cost:.4f}  "
                f"info_loss={info_loss_val:.4f}  total={total:.4f}"
            )

        if it > 0 and it % 5 == 0:
            M_mod = M.copy()
            for c in range(n_classes):
                s_idx = np.where(y_s == c)[0]
                t_idx = np.where(y_t_partial == c)[0]
                if len(t_idx) == 0:
                    continue
                M_mod[np.ix_(s_idx, t_idx)] *= (1.0 - reg_cl)
                for other_c in range(n_classes):
                    if other_c != c:
                        s_other = np.where(y_s == other_c)[0]
                        M_mod[np.ix_(s_other, t_idx)] *= (1.0 + reg_cl)

            gamma = ot.sinkhorn(a, b, M_mod, reg=reg_e, numItermax=500)

    col_sums = gamma.sum(axis=0, keepdims=True)
    col_sums[col_sums < 1e-8] = 1e-8
    X_t_adapted = (gamma / col_sums).T @ X_s
    X_t_adapted = np.nan_to_num(X_t_adapted)

    return X_t_adapted, gamma
