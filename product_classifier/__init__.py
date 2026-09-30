"""Product-category prediction from product descriptions.

Typical use:

    from product_classifier import PipelineConfig, run

    cfg = PipelineConfig()
    cfg.data.path = "data/golden_set_l1_l4.txt"
    cfg.model.n_levels = 2            # L1 + L2 only
    cfg.model.prediction_mode = "joint_path"
    result = run(cfg)
"""

from .config import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    NBConfig,
    PipelineConfig,
    SplitConfig,
    VectorizerConfig,
)
from .model import HierarchicalPathClassifier
from .pipeline import RunResult, compare_modes, run, save_outputs, sweep_depths

__all__ = [
    "DataConfig",
    "EvalConfig",
    "ModelConfig",
    "NBConfig",
    "PipelineConfig",
    "SplitConfig",
    "VectorizerConfig",
    "HierarchicalPathClassifier",
    "RunResult",
    "run",
    "save_outputs",
    "sweep_depths",
    "compare_modes",
]

__version__ = "0.1.0"
