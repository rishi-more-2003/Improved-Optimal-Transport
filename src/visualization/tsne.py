"""t-SNE visualization of domain adaptation results."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne_comparison(X_s, y_s, X_t_before, X_t_after, y_t,
                         method_name="OT", random_state=42,
                         save_path=None):
    """
    Side-by-side t-SNE plots: source, target-before, target-after adaptation.

    Parameters
    ----------
    X_s : ndarray, shape (n_source, d)
    y_s : ndarray, shape (n_source,)
    X_t_before : ndarray, shape (n_target, d)
    X_t_after : ndarray, shape (n_target, d)
    y_t : ndarray, shape (n_target,)
    method_name : str
        Label shown in the "after" panel title.
    random_state : int
    save_path : str, optional
    """
    X_all = np.vstack([X_s, X_t_before, X_t_after])
    tsne = TSNE(n_components=2, random_state=random_state)
    Z = tsne.fit_transform(X_all)

    ns = X_s.shape[0]
    nt = X_t_before.shape[0]
    Z_s = Z[:ns]
    Z_tb = Z[ns:ns + nt]
    Z_ta = Z[ns + nt:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for c in np.unique(y_s):
        axes[0].scatter(Z_s[y_s == c, 0], Z_s[y_s == c, 1],
                        label=f"Class {c}", alpha=0.7, s=15)
    axes[0].set_title("Source Domain")
    axes[0].legend()

    for c in np.unique(y_t):
        axes[1].scatter(Z_tb[y_t == c, 0], Z_tb[y_t == c, 1],
                        label=f"Class {c}", alpha=0.7, s=15)
    axes[1].set_title("Target (Before Adaptation)")
    axes[1].legend()

    for c in np.unique(y_t):
        axes[2].scatter(Z_ta[y_t == c, 0], Z_ta[y_t == c, 1],
                        label=f"Class {c}", alpha=0.7, s=15)
    axes[2].set_title(f"Target (After {method_name})")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
