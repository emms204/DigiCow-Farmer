"""
Configuration for Plan 9 — Multi-View Ensemble.

View A: E6 isotonic-style tabular (anchor).
View B: Topic-text TF-IDF/SVD + LGBM.
View C: Either (1) Node2Vec + LGBM with versioned cache (view_c_n2v_v1), or (2) PPMI-SVD + interaction TE.
Blender: constrained optimizer (max weighted score, penalty on fold variance, floor on worst-fold).
"""

from shared.constants import RANDOM_SEED

# View B: text
TFIDF_MAX_FEATURES = 500
SVD_N_COMPONENTS = 64
VIEW_B_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "verbosity": -1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# View C: backend "n2v" (Node2Vec, use cache) or "ppmi" (PPMI-SVD + interaction TE)
VIEW_C_BACKEND = "n2v"  # "n2v" | "ppmi"
# Node2Vec artifact: precompute once, then load from plan9/artifacts/view_c_n2v_{version}/
VIEW_C_N2V_ARTIFACT_VERSION = "v1"  # folder name: view_c_n2v_v1; bump when graph schema changes
VIEW_C_N2V_DIMENSIONS = 64
VIEW_C_N2V_WALK_LENGTH = 20
VIEW_C_N2V_NUM_WALKS = 10
VIEW_C_N2V_P = 1.0
VIEW_C_N2V_Q = 1.0
VIEW_C_N2V_WORKERS = 1

# View C: PPMI-SVD + interaction TE (used when VIEW_C_BACKEND == "ppmi")
VIEW_C_PPMI_DIM = 64  # TruncatedSVD components on PPMI matrix
VIEW_C_PPMI_SHIFT = 1.0  # shift for shifted PPMI (avoid log 0)
VIEW_C_TE_MIN_SUPPORT = 5  # min (entity, topic) count to use TE; else global mean
VIEW_C_TE_SHRINK = 10.0  # Bayesian shrinkage (prior count) for interaction TE
VIEW_C_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "verbosity": -1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Blender
BLEND_PENALTY_FOLD_STD = 0.5  # penalty = penalty_coef * fold_std
PROB_CLIP = (0.001, 0.999)
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 500

SEED = RANDOM_SEED
