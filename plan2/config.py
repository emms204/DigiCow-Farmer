"""
CatBoost hyperparameter configuration for Plan 2.

Uses ordered boosting for natural overfitting protection and auto
class-weight handling for the imbalanced targets.
"""

from shared.constants import RANDOM_SEED

CATBOOST_PARAMS: dict = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "iterations": 1000,
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "allow_writing_files": False,  # no temp files on disk
    "use_best_model": True,
}

EARLY_STOPPING_ROUNDS: int = 50

# Weight given to Prior samples relative to Train (dampens Prior influence)
PRIOR_SAMPLE_WEIGHT: float = 0.5
