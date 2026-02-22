"""
Configuration for Plan 5 — Simple Calibrated Model.

Logistic Regression is inherently well-calibrated, making it ideal
when Log Loss constitutes 75 % of the evaluation metric.
"""

from shared.constants import RANDOM_SEED

# ── Primary model: Elastic Net Logistic Regression ─────────────────────
ELASTIC_NET_PARAMS: dict = {
    "penalty": "elasticnet",
    "C": 0.5,
    "l1_ratio": 0.3,
    "solver": "saga",
    "max_iter": 5000,
    "random_state": RANDOM_SEED,
}

# ── Fallback model: Ridge Logistic Regression (if Elastic Net underfits) ─
RIDGE_LR_PARAMS: dict = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 3000,
    "random_state": RANDOM_SEED,
}

N_CV_FOLDS: int = 5
