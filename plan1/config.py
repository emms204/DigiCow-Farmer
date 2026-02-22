"""
LightGBM hyperparameter configuration for Plan 1.

Tuned for binary classification on imbalanced data with emphasis on
Log Loss optimisation.  Conservative settings to avoid overfitting.
"""

from shared.constants import RANDOM_SEED

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
    "is_unbalance": True,       # auto-adjust for class imbalance
    "verbosity": -1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Early stopping rounds during training with validation set
EARLY_STOPPING_ROUNDS: int = 50
NUM_BOOST_ROUND: int = 1000

