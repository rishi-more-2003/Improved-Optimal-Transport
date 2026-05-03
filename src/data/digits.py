"""MNIST -> USPS digit domain adaptation benchmark."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import rotate, zoom


def _transform_images(X, image_size=28, rotation_range=(-10, 10),
                      scale_range=(0.8, 1.2), noise_std=0.1):
    """Apply random rotation, scaling, and noise to simulate domain shift."""
    X_out = np.zeros_like(X)
    for i in range(len(X)):
        img = X[i].reshape(image_size, image_size)

        angle = np.random.uniform(*rotation_range)
        img = rotate(img, angle, reshape=False)

        scale = np.random.uniform(*scale_range)
        img = zoom(img, scale, order=1)

        h, w = img.shape
        if h > image_size or w > image_size:
            sh = max(0, (h - image_size) // 2)
            sw = max(0, (w - image_size) // 2)
            img = img[sh:sh + image_size, sw:sw + image_size]
        else:
            canvas = np.zeros((image_size, image_size))
            sh = max(0, (image_size - h) // 2)
            sw = max(0, (image_size - w) // 2)
            canvas[sh:sh + h, sw:sw + w] = img
            img = canvas

        img += np.random.normal(0, noise_std, img.shape)
        X_out[i] = np.clip(img, 0, 1).flatten()

    return X_out


def create_mnist_usps(n_samples=2000, n_pca=50, max_classes=5,
                      rotation_range=(-10, 10), scale_range=(0.8, 1.2),
                      noise_std=0.1, random_state=42):
    """
    Create an MNIST -> USPS-style domain adaptation problem.

    MNIST digits (classes 0-4) are split; the target half is augmented
    with random geometric transforms and noise to simulate USPS-like
    characteristics.  PCA reduces dimensionality before standardization.

    Parameters
    ----------
    n_samples : int
        Total samples to draw (split 50/50 source/target).
    n_pca : int
        Number of PCA components.
    max_classes : int
        Use digit classes 0..(max_classes-1).
    rotation_range, scale_range, noise_std : augmentation params.
    random_state : int

    Returns
    -------
    X_s, y_s, X_t, y_t : standardized PCA-reduced features and labels.
    X_s_raw, X_t_raw : raw 784-d pixel vectors (for sample visualization).
    """
    np.random.seed(random_state)

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    mask = y < max_classes
    X, y = X[mask], y[mask]
    idx = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[idx], y[idx]

    X = X / 255.0
    X_source_raw, X_target_raw, y_s, y_t = train_test_split(
        X, y, test_size=0.5, random_state=random_state,
    )

    X_target_raw = _transform_images(
        X_target_raw, rotation_range=rotation_range,
        scale_range=scale_range, noise_std=noise_std,
    )

    pca = PCA(n_components=n_pca)
    X_s_pca = pca.fit_transform(X_source_raw)
    X_t_pca = pca.transform(X_target_raw)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_s_pca)
    X_t = scaler.transform(X_t_pca)

    return X_s, y_s, X_t, y_t, X_source_raw, X_target_raw
