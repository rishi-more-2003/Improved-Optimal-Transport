"""
Main entry point for running all domain adaptation experiments.

Usage
-----
    python run_experiments.py              # run everything
    python run_experiments.py moons        # Two-Moons only
    python run_experiments.py digits       # MNIST -> USPS only
    python run_experiments.py semisup      # semi-supervised only
"""

import sys
import json
import os

import numpy as np

from config import ExperimentConfig
from src.data.synthetic import create_two_moons
from src.data.digits import create_mnist_usps
from src.transport.sinkhorn import sinkhorn_domain_adaptation
from src.transport.info_loss import info_loss_domain_adaptation
from src.transport.cluster_info_loss import cluster_info_loss_domain_adaptation
from src.transport.semisupervised import semisupervised_sinkhorn, semisupervised_info_loss
from src.evaluation.metrics import evaluate_adaptation

cfg = ExperimentConfig()
np.random.seed(cfg.seed)


def _header(title):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _row(method, acc):
    print(f"  {method:<35s} {acc:6.1f}%")


def run_two_moons():
    """Run all methods on the Two-Moons benchmark."""
    _header("Two-Moons Domain Adaptation")
    sc = cfg.synthetic
    tc = cfg.transport

    X_s, y_s, X_t, y_t, _, _ = create_two_moons(
        n_samples=sc.n_samples, noise_level=sc.noise_level,
        rotation_deg=sc.rotation_deg, scale_factor=sc.scale_factor,
        random_state=sc.random_state,
    )

    results = {}

    # No adaptation
    acc = evaluate_adaptation(X_s, y_s, X_t, y_t)
    results["No Adaptation"] = acc
    _row("No Adaptation", acc * 100)

    # Sinkhorn OT
    X_adapted, _ = sinkhorn_domain_adaptation(X_s, y_s, X_t, reg_e=tc.reg_e)
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["Sinkhorn OT"] = acc
    _row("Sinkhorn OT", acc * 100)

    # Cluster OT + Info-Loss (Ours)
    X_adapted, _ = cluster_info_loss_domain_adaptation(
        X_s, y_s, X_t, reg_e=tc.reg_e, alpha=tc.alpha,
        n_clusters=tc.n_clusters, numItermax=tc.numItermax, verbose=True,
    )
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["Cluster OT + Info-Loss"] = acc
    _row("Cluster OT + Info-Loss (Ours)", acc * 100)

    # OT + Info-Loss (Ours)
    X_adapted, _ = info_loss_domain_adaptation(
        X_s, y_s, X_t, reg_e=tc.reg_e, alpha=tc.alpha,
        numItermax=tc.numItermax, verbose=True,
    )
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["OT + Info-Loss"] = acc
    _row("OT + Info-Loss (Ours)", acc * 100)

    return results


def run_digits():
    """Run all methods on the MNIST -> USPS benchmark."""
    _header("MNIST -> USPS Domain Adaptation")
    dc = cfg.digits
    tc = cfg.transport

    X_s, y_s, X_t, y_t, _, _ = create_mnist_usps(
        n_samples=dc.n_samples, n_pca=dc.n_pca_components,
        max_classes=dc.max_classes, random_state=dc.random_state,
    )

    results = {}

    acc = evaluate_adaptation(X_s, y_s, X_t, y_t)
    results["No Adaptation"] = acc
    _row("No Adaptation", acc * 100)

    X_adapted, _ = sinkhorn_domain_adaptation(X_s, y_s, X_t, reg_e=tc.reg_e)
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["Sinkhorn OT"] = acc
    _row("Sinkhorn OT", acc * 100)

    X_adapted, _ = cluster_info_loss_domain_adaptation(
        X_s, y_s, X_t, reg_e=tc.reg_e, alpha=tc.alpha,
        n_clusters=tc.n_clusters, numItermax=tc.numItermax, verbose=True,
    )
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["Cluster OT + Info-Loss"] = acc
    _row("Cluster OT + Info-Loss (Ours)", acc * 100)

    X_adapted, _ = info_loss_domain_adaptation(
        X_s, y_s, X_t, reg_e=tc.reg_e, alpha=tc.alpha,
        numItermax=tc.numItermax, verbose=True,
    )
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["OT + Info-Loss"] = acc
    _row("OT + Info-Loss (Ours)", acc * 100)

    return results


def run_semisupervised():
    """Run semi-supervised variants on Two-Moons."""
    _header("Semi-Supervised Domain Adaptation (Two-Moons)")
    sc = cfg.synthetic
    tc = cfg.transport

    X_s, y_s, X_t, y_t, _, _ = create_two_moons(
        n_samples=sc.n_samples // 2, noise_level=sc.noise_level,
        rotation_deg=sc.rotation_deg, scale_factor=sc.scale_factor,
        random_state=sc.random_state,
    )

    n_labeled = max(1, int(tc.label_fraction * len(y_t)))
    labeled_idx = np.random.choice(len(y_t), n_labeled, replace=False)
    y_t_partial = np.full_like(y_t, -1)
    y_t_partial[labeled_idx] = y_t[labeled_idx]

    results = {}

    acc = evaluate_adaptation(X_s, y_s, X_t, y_t)
    results["No Adaptation"] = acc
    _row("No Adaptation", acc * 100)

    X_adapted, _ = sinkhorn_domain_adaptation(X_s, y_s, X_t, reg_e=tc.reg_e)
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["Unsupervised Sinkhorn"] = acc
    _row("Unsupervised Sinkhorn OT", acc * 100)

    X_adapted, _ = semisupervised_sinkhorn(
        X_s, y_s, X_t, y_t_partial, reg_e=tc.reg_e, reg_cl=tc.reg_cl,
    )
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["Semi-Sup Sinkhorn"] = acc
    _row("Semi-Sup Sinkhorn OT", acc * 100)

    X_adapted, _ = semisupervised_info_loss(
        X_s, y_s, X_t, y_t_partial, reg_e=tc.reg_e, reg_cl=tc.reg_cl,
        alpha=tc.alpha, numItermax=tc.numItermax, verbose=True,
    )
    acc = evaluate_adaptation(X_s, y_s, X_adapted, y_t)
    results["Semi-Sup Info-Loss"] = acc
    _row("Semi-Sup Info-Loss (Ours)", acc * 100)

    return results


def main():
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["moons", "digits", "semisup"]
    all_results = {}

    for task in tasks:
        if task == "moons":
            all_results["two_moons"] = run_two_moons()
        elif task == "digits":
            all_results["mnist_usps"] = run_digits()
        elif task == "semisup":
            all_results["semisupervised"] = run_semisupervised()
        else:
            print(f"Unknown task: {task}  (choose from: moons, digits, semisup)")

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
