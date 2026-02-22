#!/usr/bin/env python3
"""
E2 Dual AUC/LL experiment: per target train AUC-optimized and LL-optimized models,
write separate columns; compare to E0 single-output baseline.

Hypothesis E2: Separate AUC/LL outputs beat single-output compromise.
Experiment: Per target: train AUC-optimized and LL-optimized models, write separate columns.
Primary pass metric: Weighted score.
Guardrails: 7d LL protected (Δ7dLL <= 0).
Pass threshold: ΔWeighted >= +0.004 with Δ7dLL <= 0.
If fail: Keep single-output columns.

Usage:
    python e2_dual_auc_ll.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.constants import DATE_COL, TARGET_COLS
from shared.data_loader import DataLoader
from shared.evaluation import calculate_weighted_score
from shared.feature_engineering import MINIMAL_FEATURE_GROUPS, FeatureEngineer
from sklearn.metrics import log_loss, roc_auc_score

from e0_reproducibility import (
    E0_CLIP,
    E0_MIN_TRAIN_SIZE,
    E0_N_SPLITS,
    E0_SEED,
    rolling_forward_splits,
    run_single_e0_harness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# E2 pass thresholds
E2_DWEIGHTED_MIN = 0.004   # weighted score must improve by at least this
E2_D7D_LL_MAX = 0.0       # 7d LogLoss must not increase (guardrail)


def run_single_e2_harness(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> dict[str, Any]:
    """
    One run of E2: per target train AUC-optimized and LL-optimized LGBM;
    fill OOF for AUC column and LL column per target; compute weighted score from 6 metrics.
    """
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    params_ll = {**LGBM_PARAMS, "metric": "binary_logloss", "random_state": seed}
    params_auc = {**LGBM_PARAMS, "metric": "auc", "random_state": seed}

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    oof_auc = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_ll = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)

    for split in splits:
        fold = split["fold"]
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_cutoff = split["train_cutoff"]

        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(MINIMAL_FEATURE_GROUPS)
        fe.fit(train_df.iloc[train_pos])

        X_tr = fe.transform(train_df.iloc[train_pos])
        X_va = fe.transform(train_df.iloc[val_pos])

        for target in TARGET_COLS:
            y_tr = train_df[target].iloc[train_pos].values
            y_va = train_df[target].iloc[val_pos].values

            # AUC-optimized model (metric=auc)
            train_set_a = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set_a = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster_a = lgb.train(
                params=params_auc,
                train_set=train_set_a,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set_a, val_set_a],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_auc_va = booster_a.predict(X_va, num_iteration=booster_a.best_iteration)
            pred_auc_va = np.clip(pred_auc_va, *E0_CLIP)
            oof_auc[target][val_pos] = pred_auc_va

            # LL-optimized model (metric=binary_logloss)
            train_set_l = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set_l = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster_l = lgb.train(
                params=params_ll,
                train_set=train_set_l,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set_l, val_set_l],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_ll_va = booster_l.predict(X_va, num_iteration=booster_l.best_iteration)
            pred_ll_va = np.clip(pred_ll_va, *E0_CLIP)
            oof_ll[target][val_pos] = pred_ll_va

            oof_mask[val_pos] = True

        logger.info(
            "  Fold %d: train=%d val=%d (dual AUC/LL per target)",
            fold, len(train_pos), len(val_pos),
        )

    # OOF: 6 metrics (AUC column for AUC, LL column for LogLoss)
    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]
    p_07_auc = oof_auc[TARGET_COLS[0]][oof_mask]
    p_07_ll = oof_ll[TARGET_COLS[0]][oof_mask]
    p_90_auc = oof_auc[TARGET_COLS[1]][oof_mask]
    p_90_ll = oof_ll[TARGET_COLS[1]][oof_mask]
    p_120_auc = oof_auc[TARGET_COLS[2]][oof_mask]
    p_120_ll = oof_ll[TARGET_COLS[2]][oof_mask]

    auc_07 = roc_auc_score(y_07, p_07_auc) if len(np.unique(y_07)) > 1 else 0.5
    ll_07 = log_loss(y_07, p_07_ll)
    auc_90 = roc_auc_score(y_90, p_90_auc) if len(np.unique(y_90)) > 1 else 0.5
    ll_90 = log_loss(y_90, p_90_ll)
    auc_120 = roc_auc_score(y_120, p_120_auc) if len(np.unique(y_120)) > 1 else 0.5
    ll_120 = log_loss(y_120, p_120_ll)

    weighted_score = calculate_weighted_score(auc_07, ll_07, auc_90, ll_90, auc_120, ll_120)

    return {
        "weighted_score": weighted_score,
        "oof_07_ll": ll_07,
        "oof_07_auc": auc_07,
        "oof_90_ll": ll_90,
        "oof_90_auc": auc_90,
        "oof_120_ll": ll_120,
        "oof_120_auc": auc_120,
    }


def get_e2_oof(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run E2 harness (LL column per target) and return OOF (p07, p90, p120) and oof_mask. For E6 blending."""
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    params_ll = {**LGBM_PARAMS, "metric": "binary_logloss", "random_state": seed}

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    oof_ll = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)

    for split in splits:
        train_pos = split["train_pos"]
        val_pos = split["val_pos"]
        train_cutoff = split["train_cutoff"]

        prior_asof = prior_df[prior_df[DATE_COL] < train_cutoff].copy()
        fe = FeatureEngineer(prior_asof)
        fe.set_feature_groups(MINIMAL_FEATURE_GROUPS)
        fe.fit(train_df.iloc[train_pos])

        X_tr = fe.transform(train_df.iloc[train_pos])
        X_va = fe.transform(train_df.iloc[val_pos])

        for target in TARGET_COLS:
            y_tr = train_df[target].iloc[train_pos].values
            y_va = train_df[target].iloc[val_pos].values
            train_set_l = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set_l = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster_l = lgb.train(
                params=params_ll,
                train_set=train_set_l,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set_l, val_set_l],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_ll_va = booster_l.predict(X_va, num_iteration=booster_l.best_iteration)
            oof_ll[target][val_pos] = np.clip(pred_ll_va, *E0_CLIP)
            oof_mask[val_pos] = True

    return (
        oof_ll[TARGET_COLS[0]],
        oof_ll[TARGET_COLS[1]],
        oof_ll[TARGET_COLS[2]],
        oof_mask,
    )


def main() -> None:
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E2 DUAL AUC/LL EXPERIMENT")
    logger.info("  Pass: ΔWeighted >= %.3f AND Δ7dLL <= %.1f (7d LL protected)", E2_DWEIGHTED_MIN, E2_D7D_LL_MAX)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    # Baseline (E0 single-output)
    logger.info("\n--- E0 baseline (single-output: one pred per target for both AUC & LL) ---")
    baseline = run_single_e0_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    logger.info("  Baseline weighted=%.6f  7d_ll=%.6f  7d_auc=%.6f",
                baseline["weighted_score"], baseline["oof_07_ll"], baseline["oof_07_auc"])

    # E2 (dual AUC/LL columns)
    logger.info("\n--- E2 (separate AUC-optimized and LL-optimized models per target) ---")
    e2 = run_single_e2_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    logger.info("  E2     weighted=%.6f  7d_ll=%.6f  7d_auc=%.6f (from separate columns)",
                e2["weighted_score"], e2["oof_07_ll"], e2["oof_07_auc"])

    d_weighted = e2["weighted_score"] - baseline["weighted_score"]
    d_7d_ll = e2["oof_07_ll"] - baseline["oof_07_ll"]

    pass_weighted = d_weighted >= E2_DWEIGHTED_MIN
    pass_7d_ll = d_7d_ll <= E2_D7D_LL_MAX
    primary_pass = pass_weighted and pass_7d_ll

    logger.info("\n" + "=" * 70)
    logger.info("E2 RESULT")
    logger.info("  ΔWeighted = %.6f  (need >= %.3f)  %s", d_weighted, E2_DWEIGHTED_MIN, "OK" if pass_weighted else "FAIL")
    logger.info("  Δ7dLL     = %.6f  (need <= %.1f, 7d LL protected)  %s", d_7d_ll, E2_D7D_LL_MAX, "OK" if pass_7d_ll else "FAIL")
    if primary_pass:
        logger.info("  PASS: Adopt separate AUC/LL columns.")
    else:
        logger.warning("  FAIL: Keep single-output columns.")
    logger.info("=" * 70)

    sys.exit(0 if primary_pass else 1)


if __name__ == "__main__":
    main()
