"""
Configuration for Plan 3 — Stacking Ensemble.

Defines hyperparameters for each base learner and the meta-learner.
Base learners are intentionally diverse (different algorithms, different
regularisation) to maximise ensemble benefit.
"""

from shared.constants import RANDOM_SEED

# ── LightGBM base learner ─────────────────────────────────────────────
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

# ── XGBoost base learner ──────────────────────────────────────────────
XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,  # handled per-target at runtime
    "verbosity": 0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ── CatBoost base learner ─────────────────────────────────────────────
CATBOOST_PARAMS: dict = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "iterations": 800,
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "allow_writing_files": False,
    "use_best_model": True,
}

# ── Random Forest base learner ─────────────────────────────────────────
RF_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ── Logistic Regression base learner ───────────────────────────────────
LR_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 5000,
    "solver": "saga",
    "random_state": RANDOM_SEED,
}

# ── Meta-learner ──────────────────────────────────────────────────────
META_LR_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}

# ── Training settings ─────────────────────────────────────────────────
NUM_BOOST_ROUND: int = 800
EARLY_STOPPING_ROUNDS: int = 50
N_CV_FOLDS: int = 5
