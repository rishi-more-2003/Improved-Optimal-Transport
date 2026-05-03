"""
Generate publication-quality figures from experiment results.

Usage
-----
    python analyze_results.py                    # all figures
    python analyze_results.py moons              # Two-Moons only
    python analyze_results.py digits             # MNIST -> USPS only
    python analyze_results.py semisup            # semi-supervised only
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from config import ExperimentConfig
from src.data.synthetic import create_two_moons
from src.data.digits import create_mnist_usps
from src.transport.sinkhorn import sinkhorn_domain_adaptation
from src.transport.info_loss import info_loss_domain_adaptation
from src.transport.cluster_info_loss import cluster_info_loss_domain_adaptation
from src.transport.semisupervised import semisupervised_sinkhorn, semisupervised_info_loss
from src.evaluation.metrics import evaluate_adaptation
from src.visualization.tsne import plot_tsne_comparison
from src.visualization.transport_plan import plot_transport_plan

cfg = ExperimentConfig()
np.random.seed(cfg.seed)
os.makedirs(cfg.output_dir, exist_ok=True)


def figures_two_moons():
    """Generate all Two-Moons figures."""
    print("Generating Two-Moons figures ...")
    sc, tc = cfg.synthetic, cfg.transport

    X_s, y_s, X_t, y_t, X_s_raw, X_t_raw = create_two_moons(
        n_samples=sc.n_samples, noise_level=sc.noise_level,
        rotation_deg=sc.rotation_deg, scale_factor=sc.scale_factor,
        random_state=sc.random_state,
    )

    methods = {
        "Sinkhorn OT": sinkhorn_domain_adaptation(X_s, y_s, X_t, reg_e=tc.reg_e),
        "OT + Info-Loss": info_loss_domain_adaptation(
            X_s, y_s, X_t, reg_e=tc.reg_e, alpha=tc.alpha, numItermax=tc.numItermax,
        ),
        "Cluster OT + Info-Loss": cluster_info_loss_domain_adaptation(
            X_s, y_s, X_t, reg_e=tc.reg_e, alpha=tc.alpha,
            n_clusters=tc.n_clusters, numItermax=tc.numItermax,
        ),
    }

    for name, (X_adapted, gamma) in methods.items():
        tag = name.lower().replace(" ", "_").replace("+", "").replace("-", "")
        plot_tsne_comparison(
            X_s, y_s, X_t, X_adapted, y_t, method_name=name,
            save_path=os.path.join(cfg.output_dir, f"moons_tsne_{tag}.png"),
        )

    _, gamma_sink = methods["Sinkhorn OT"]
    plot_transport_plan(
        X_s, X_t, gamma_sink, y_s=y_s, top_k=3,
        title="Sinkhorn OT Transport Plan",
        save_path=os.path.join(cfg.output_dir, "moons_transport_sinkhorn.png"),
    )

    # Domain visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in np.unique(y_s):
        axes[0].scatter(X_s_raw[y_s == c, 0], X_s_raw[y_s == c, 1],
                        label=f"Class {c}", alpha=0.8, s=20, edgecolors="k")
    axes[0].set_title("Source Domain")
    axes[0].legend()
    for c in np.unique(y_t):
        axes[1].scatter(X_t_raw[y_t == c, 0], X_t_raw[y_t == c, 1],
                        label=f"Class {c}", alpha=0.8, s=20, edgecolors="k")
    axes[1].set_title("Target Domain")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "moons_domains.png"),
                dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.show()

    # Bar chart comparison
    labels, accs = [], []
    acc_none = evaluate_adaptation(X_s, y_s, X_t, y_t)
    labels.append("No Adapt.")
    accs.append(acc_none * 100)
    for name, (X_adapted, _) in methods.items():
        acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t) * 100
        labels.append(name)
        accs.append(acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#bdc3c7", "#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, accs, color=colors[:len(labels)], edgecolor="k")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Target Accuracy (%)")
    ax.set_title("Two-Moons: Method Comparison")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "moons_bar_comparison.png"),
                dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.show()

    print("  Done.")


def figures_digits():
    """Generate all MNIST -> USPS figures."""
    print("Generating MNIST -> USPS figures ...")
    dc, tc = cfg.digits, cfg.transport

    X_s, y_s, X_t, y_t, X_s_raw, X_t_raw = create_mnist_usps(
        n_samples=dc.n_samples, n_pca=dc.n_pca_components,
        max_classes=dc.max_classes, random_state=dc.random_state,
    )

    methods = {
        "Sinkhorn OT": sinkhorn_domain_adaptation(X_s, y_s, X_t, reg_e=tc.reg_e),
        "OT + Info-Loss": info_loss_domain_adaptation(
            X_s, y_s, X_t, reg_e=tc.reg_e, alpha=tc.alpha, numItermax=tc.numItermax,
        ),
        "Cluster OT + Info-Loss": cluster_info_loss_domain_adaptation(
            X_s, y_s, X_t, reg_e=tc.reg_e, alpha=tc.alpha,
            n_clusters=tc.n_clusters, numItermax=tc.numItermax,
        ),
    }

    for name, (X_adapted, _) in methods.items():
        tag = name.lower().replace(" ", "_").replace("+", "").replace("-", "")
        plot_tsne_comparison(
            X_s, y_s, X_t, X_adapted, y_t, method_name=name,
            save_path=os.path.join(cfg.output_dir, f"digits_tsne_{tag}.png"),
        )

    # Sample digit visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        axes[0, i].imshow(X_s_raw[i].reshape(28, 28), cmap="gray")
        axes[0, i].set_title(f"Src: {y_s[i]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(X_t_raw[i].reshape(28, 28), cmap="gray")
        axes[1, i].set_title(f"Tgt: {y_t[i]}")
        axes[1, i].axis("off")
    plt.suptitle("Sample Digits: Source (MNIST) vs Target (Transformed)")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "digits_samples.png"),
                dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.show()

    # Bar chart
    labels, accs = [], []
    acc_none = evaluate_adaptation(X_s, y_s, X_t, y_t)
    labels.append("No Adapt.")
    accs.append(acc_none * 100)
    for name, (X_adapted, _) in methods.items():
        acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t) * 100
        labels.append(name)
        accs.append(acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#bdc3c7", "#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, accs, color=colors[:len(labels)], edgecolor="k")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Target Accuracy (%)")
    ax.set_title("MNIST -> USPS: Method Comparison")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "digits_bar_comparison.png"),
                dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.show()

    print("  Done.")


def figures_semisupervised():
    """Generate semi-supervised comparison figures."""
    print("Generating semi-supervised figures ...")
    sc, tc = cfg.synthetic, cfg.transport

    X_s, y_s, X_t, y_t, _, _ = create_two_moons(
        n_samples=sc.n_samples // 2, noise_level=sc.noise_level,
        rotation_deg=sc.rotation_deg, scale_factor=sc.scale_factor,
        random_state=sc.random_state,
    )

    n_labeled = max(1, int(tc.label_fraction * len(y_t)))
    labeled_idx = np.random.choice(len(y_t), n_labeled, replace=False)
    y_t_partial = np.full_like(y_t, -1)
    y_t_partial[labeled_idx] = y_t[labeled_idx]

    X_unsup, _ = sinkhorn_domain_adaptation(X_s, y_s, X_t, reg_e=tc.reg_e)
    X_semisup_sink, _ = semisupervised_sinkhorn(
        X_s, y_s, X_t, y_t_partial, reg_e=tc.reg_e, reg_cl=tc.reg_cl,
    )
    X_semisup_info, _ = semisupervised_info_loss(
        X_s, y_s, X_t, y_t_partial, reg_e=tc.reg_e, reg_cl=tc.reg_cl,
        alpha=tc.alpha, numItermax=tc.numItermax,
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for c in np.unique(y_s):
        axes[0].scatter(X_s[y_s == c, 0], X_s[y_s == c, 1],
                        label=f"Class {c}", alpha=0.7)
    axes[0].set_title("Source")
    axes[0].legend()

    for c in np.unique(y_t):
        axes[1].scatter(X_unsup[y_t == c, 0], X_unsup[y_t == c, 1],
                        label=f"Class {c}", alpha=0.7)
    axes[1].set_title("Unsupervised Sinkhorn")
    axes[1].legend()

    for c in np.unique(y_t):
        axes[2].scatter(X_semisup_sink[y_t == c, 0], X_semisup_sink[y_t == c, 1],
                        label=f"Class {c}", alpha=0.7)
    axes[2].set_title("Semi-Sup Sinkhorn")
    axes[2].legend()

    for c in np.unique(y_t):
        axes[3].scatter(X_semisup_info[y_t == c, 0], X_semisup_info[y_t == c, 1],
                        label=f"Class {c}", alpha=0.7)
    axes[3].set_title("Semi-Sup Info-Loss (Ours)")
    axes[3].legend()

    plt.suptitle(f"Semi-Supervised Comparison ({tc.label_fraction*100:.0f}% labeled)")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "semisup_comparison.png"),
                dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.show()

    print("  Done.")


def main():
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["moons", "digits", "semisup"]
    for task in tasks:
        if task == "moons":
            figures_two_moons()
        elif task == "digits":
            figures_digits()
        elif task == "semisup":
            figures_semisupervised()
        else:
            print(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
