"""
CatBoost hyperparameter configuration for Plan 2.

Two param sets for dual optimisation (LogLoss vs AUC columns):
  - CATBOOST_PARAMS: LogLoss-optimised (conservative).
  - CATBOOST_AUC_PARAMS: AUC-optimised (more capacity for ranking).
"""

from shared.constants import RANDOM_SEED

# LogLoss columns: conservative, well-calibrated
CATBOOST_PARAMS: dict = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "iterations": 1000,
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "allow_writing_files": False,
    "use_best_model": True,
}

# AUC columns: more capacity for ranking
CATBOOST_AUC_PARAMS: dict = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.06,
    "depth": 7,
    "l2_leaf_reg": 1.0,
    "iterations": 1000,
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "allow_writing_files": False,
    "use_best_model": True,
}

EARLY_STOPPING_ROUNDS: int = 50

# Weight given to Prior samples relative to Train (dampens Prior influence)
PRIOR_SAMPLE_WEIGHT: float = 0.5
