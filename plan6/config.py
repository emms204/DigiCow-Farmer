"""
Configuration for Plan 6 — hazard-style neural network.
"""

from shared.constants import (
    CV_STRATEGY_DEFAULT,
    CV_TIME_CUTOFF_DEFAULT,
    RANDOM_SEED,
)

# Feature groups for Plan 6.
# Start from the robust minimal signal and avoid unstable groups by default.
FEATURE_GROUPS: dict[str, bool] = {
    "direct_features": True,
    "date_features": False,
    "topic_features": True,
    "trainer_features": False,
    "prior_farmer_history": False,
    "aggregation_features": False,
    "confidence_aggregates": False,
    "interactions": False,
    "recency_intensity": False,
    "quality_features": False,
}

# CV settings
N_CV_FOLDS: int = 5
CV_STRATEGY: str = CV_STRATEGY_DEFAULT
CV_TIME_CUTOFF: str = CV_TIME_CUTOFF_DEFAULT

# Neural network architecture/training
HIDDEN_DIMS: tuple[int, ...] = (128, 64)
DROPOUT: float = 0.2
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 512
EPOCHS: int = 120
EARLY_STOPPING_PATIENCE: int = 15
MIN_DELTA: float = 1e-4

# Hazard interval loss weights.
# 7-day dominates final metric, so we emphasize the first interval.
INTERVAL_LOSS_WEIGHTS: tuple[float, float, float] = (0.8, 0.1, 0.1)

# Positive-class weighting cap for BCEWithLogits loss.
MAX_POS_WEIGHT: float = 25.0

# Calibration for cumulative outputs.
# "none" or "platt"
CALIBRATION: str = "platt"

# Reproducibility
SEED: int = RANDOM_SEED

