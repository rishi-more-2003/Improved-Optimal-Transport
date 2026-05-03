"""Centralized hyperparameters for all experiments."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SyntheticDataConfig:
    n_samples: int = 500
    noise_level: float = 0.1
    rotation_deg: float = 45.0
    scale_factor: float = 1.5
    random_state: int = 42


@dataclass
class DigitsDataConfig:
    n_samples: int = 2000
    n_pca_components: int = 50
    max_classes: int = 5
    rotation_range: tuple = (-10, 10)
    scale_range: tuple = (0.8, 1.2)
    noise_std: float = 0.1
    random_state: int = 42


@dataclass
class TransportConfig:
    reg_e: float = 0.01
    reg_cl: float = 0.1
    alpha: float = 1.0
    n_clusters: int = 5
    numItermax: int = 50
    sinkhorn_max_iter: int = 1000
    label_fraction: float = 0.1


@dataclass
class VisualizationConfig:
    figsize_single: tuple = (10, 8)
    figsize_triple: tuple = (18, 6)
    figsize_quad: tuple = (18, 6)
    tsne_random_state: int = 42
    dpi: int = 150
    save_format: str = "png"


@dataclass
class ExperimentConfig:
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    digits: DigitsDataConfig = field(default_factory=DigitsDataConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output_dir: str = "figures"
    seed: int = 42
