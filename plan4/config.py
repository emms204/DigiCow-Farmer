"""
Configuration for Plan 4 — Dual Optimisation.

Two separate LightGBM parameter sets:
    - ``LOGLOSS_PARAMS``: conservative, high-regularisation for calibration.
    - ``AUC_PARAMS``: aggressive, more capacity for ranking.
"""

from shared.constants import RANDOM_SEED

# ── Model A: optimised for Log Loss (75 % weight) ─────────────────────
# Conservative: fewer leaves, more regularisation, slower learning rate.
LOGLOSS_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 20,
    "max_depth": 5,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "is_unbalance": True,
    "verbosity": -1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ── Model B: optimised for ROC-AUC (25 % weight) ──────────────────────
# Aggressive: more leaves, less regularisation, focus on ranking.
AUC_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.05,
    "lambda_l2": 0.5,
    "is_unbalance": True,
    "verbosity": -1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

NUM_BOOST_ROUND: int = 1000
EARLY_STOPPING_ROUNDS: int = 50
N_CV_FOLDS: int = 5

