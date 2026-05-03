"""Synthetic Two-Moons domain adaptation benchmark."""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def create_two_moons(n_samples=500, noise_level=0.1, rotation_deg=45.0,
                     scale_factor=1.5, random_state=42):
    """
    Generate a Two-Moons domain adaptation problem.

    The source domain is the standard scikit-learn moons dataset.
    The target domain is transformed by rotation and scaling to
    induce a controlled domain shift.

    Parameters
    ----------
    n_samples : int
        Samples per domain.
    noise_level : float
        Gaussian noise added to the moons.
    rotation_deg : float
        Rotation angle (degrees) applied to the target domain.
    scale_factor : float
        Uniform scaling applied to the target domain.
    random_state : int

    Returns
    -------
    X_s, y_s : source features and labels (standardized).
    X_t, y_t : target features and labels (standardized).
    X_s_raw, X_t_raw : un-standardized features (for visualization).
    """
    X_s, y_s = datasets.make_moons(n_samples=n_samples, noise=noise_level,
                                   random_state=random_state)
    X_t, y_t = datasets.make_moons(n_samples=n_samples, noise=noise_level,
                                   random_state=random_state + 1)

    theta = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    X_t = X_t @ R.T * scale_factor

    scaler = StandardScaler()
    X_s_scaled = scaler.fit_transform(X_s)
    X_t_scaled = scaler.transform(X_t)

    return X_s_scaled, y_s, X_t_scaled, y_t, X_s, X_t
