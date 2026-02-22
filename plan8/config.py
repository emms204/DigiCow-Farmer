"""
Configuration for Plan 8 — Cohort-Aware Mixture + Multi-Calibration.

Two experts (warm vs cold), gating on stable features, cohort-wise calibration.
"""

from shared.constants import RANDOM_SEED

# Warm expert: prior/history/context (full context for farmers with history).
WARM_FEATURE_GROUPS: dict[str, bool] = {
    "direct_features": True,
    "date_features": True,
    "topic_features": True,
    "trainer_features": True,
    "prior_farmer_history": True,
    "aggregation_features": True,
    "confidence_aggregates": False,
    "interactions": False,
    "recency_intensity": False,
    "quality_features": False,
}

# Cold expert: no prior-history block; minimal + topic + stable context only.
COLD_FEATURE_GROUPS: dict[str, bool] = {
    "direct_features": True,
    "date_features": True,
    "topic_features": True,
    "trainer_features": False,
    "prior_farmer_history": False,
    "aggregation_features": False,
    "confidence_aggregates": False,
    "interactions": False,
    "recency_intensity": False,
    "quality_features": False,
}

# LGBM (reuse Plan 1 style)
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "verbosity": -1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
EARLY_STOPPING_ROUNDS: int = 50
NUM_BOOST_ROUND: int = 1000

# Cohort calibration: choose one of these per cohort by OOF LL.
CALIBRATOR_OPTIONS: list[str] = ["none", "platt", "isotonic"]

# Probability clip (same as E0)
PROB_CLIP = (0.001, 0.999)

SEED: int = RANDOM_SEED
