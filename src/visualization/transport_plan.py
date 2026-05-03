"""Visualization of optimal transport plans."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_transport_plan(X_s, X_t, T, y_s=None, top_k=3,
                        title="OT Transport Plan", ax=None, save_path=None):
    """
    Visualize the transport plan by drawing links between source and target.

    Parameters
    ----------
    X_s, X_t : ndarray, shape (n, 2)
        Source and target 2-D features.
    T : ndarray, shape (n_source, n_target)
        Transport matrix.
    y_s : ndarray, optional
        Source labels (used for coloring).
    top_k : int
        Number of strongest links to draw per source point.
    title : str
    ax : matplotlib Axes, optional
    save_path : str, optional
        If provided, save figure to this path.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(T.shape[0]):
        top_indices = np.argsort(T[i])[-top_k:]
        for j in top_indices:
            alpha = min(T[i, j] * 100, 1.0)
            ax.plot(
                [X_s[i, 0], X_t[j, 0]],
                [X_s[i, 1], X_t[j, 1]],
                color="gray", alpha=alpha, linewidth=0.5,
            )

    if y_s is not None:
        colors = cm.viridis(y_s / (y_s.max() + 1e-9))
    else:
        colors = "steelblue"

    ax.scatter(X_s[:, 0], X_s[:, 1], c=colors, label="Source",
               s=40, edgecolor="k", alpha=0.9, zorder=3)
    ax.scatter(X_t[:, 0], X_t[:, 1], c="orange", label="Target",
               s=40, edgecolor="k", alpha=0.7, zorder=3)

    ax.set_title(title)
    ax.legend()

    if save_path and own_fig:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if own_fig:
        plt.tight_layout()
        plt.show()
