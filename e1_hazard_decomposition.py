#!/usr/bin/env python3
"""
E1 Hazard decomposition experiment: interval models (<=7, 8-90, 91-120) with GBDT;
reconstruct cumulative probs; compare to E0 baseline (direct cumulative targets).

Hypothesis E1: Hazard decomposition improves consistency and LogLoss.
Experiment: Train interval models; reconstruct P7, P90, P120; same harness as E0.
Primary pass metric: 7d LogLoss + weighted score.
Guardrails: AUC must not collapse; monotonicity guaranteed by construction.
Pass threshold: ΔWeighted >= +0.005 AND Δ7dLL <= -0.003.
If fail: Revert to cumulative direct targets.

Usage:
    python e1_hazard_decomposition.py
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

from shared.constants import (
    DATE_COL,
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
    RANDOM_SEED,
    TARGET_COLS,
)
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

# E1 pass thresholds
E1_DWEIGHTED_MIN = 0.005   # weighted score must improve by at least this
E1_D7D_LL_MAX = -0.003     # 7d LogLoss must decrease by at least 0.003 (i.e. delta <= -0.003)
E1_AUC_COLLAPSE_TOL = 0.02 # guardrail: no AUC drop worse than this vs baseline


def run_single_e1_harness(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> dict[str, Any]:
    """
    One run of E1: interval targets y1=[0,7], y2=(7,90], y3=(90,120];
    train 3 LGBMs per fold; reconstruct P7, P90, P120; clip; return OOF metrics.
    """
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    # Interval targets from cumulative
    y07 = train_df[TARGET_COLS[0]].values  # adopted_within_07_days
    y90 = train_df[TARGET_COLS[1]].values  # adopted_within_90_days
    y120 = train_df[TARGET_COLS[2]].values  # adopted_within_120_days
    y1 = y07
    y2 = ((y90 == 1) & (y07 == 0)).astype(np.float64)
    y3 = ((y120 == 1) & (y90 == 0)).astype(np.float64)

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    oof_p7 = np.zeros(n_rows, dtype=float)
    oof_p90 = np.zeros(n_rows, dtype=float)
    oof_p120 = np.zeros(n_rows, dtype=float)
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

        # Train 3 interval models
        for name, y_tr_vals, y_va_vals in [
            ("h1", y1[train_pos], y1[val_pos]),
            ("h2", y2[train_pos], y2[val_pos]),
            ("h3", y3[train_pos], y3[val_pos]),
        ]:
            train_set = lgb.Dataset(X_tr, label=y_tr_vals, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va_vals, free_raw_data=False)
            callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
            booster = lgb.train(
                params={**LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )
            h = booster.predict(X_va, num_iteration=booster.best_iteration)
            h = np.clip(h, *E0_CLIP)
            if name == "h1":
                h1_va = h
            elif name == "h2":
                h2_va = h
            else:
                h3_va = h

        # Reconstruct cumulative (monotonic by construction)
        p7_va = h1_va
        p90_va = 1.0 - (1.0 - h1_va) * (1.0 - h2_va)
        p120_va = 1.0 - (1.0 - h1_va) * (1.0 - h2_va) * (1.0 - h3_va)
        p7_va = np.clip(p7_va, *E0_CLIP)
        p90_va = np.clip(p90_va, *E0_CLIP)
        p120_va = np.clip(p120_va, *E0_CLIP)

        oof_p7[val_pos] = p7_va
        oof_p90[val_pos] = p90_va
        oof_p120[val_pos] = p120_va
        oof_mask[val_pos] = True

        logger.info(
            "  Fold %d: train=%d val=%d (P7≤P90≤P120: %s)",
            fold, len(train_pos), len(val_pos),
            np.all(p7_va <= p90_va) and np.all(p90_va <= p120_va),
        )

    # OOF metrics (evaluate reconstructed cumulatives vs cumulative labels)
    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]
    p_07 = oof_p7[oof_mask]
    p_90 = oof_p90[oof_mask]
    p_120 = oof_p120[oof_mask]

    # Monotonicity guardrail
    monotonic_ok = np.all(p_07 <= p_90) and np.all(p_90 <= p_120)
    if not monotonic_ok:
        logger.warning("  Guardrail: monotonicity violated on OOF (should not happen by construction)")

    oof_07_ll = log_loss(y_07, p_07)
    oof_07_auc = roc_auc_score(y_07, p_07) if len(np.unique(y_07)) > 1 else 0.5
    oof_90_ll = log_loss(y_90, p_90)
    oof_90_auc = roc_auc_score(y_90, p_90) if len(np.unique(y_90)) > 1 else 0.5
    oof_120_ll = log_loss(y_120, p_120)
    oof_120_auc = roc_auc_score(y_120, p_120) if len(np.unique(y_120)) > 1 else 0.5

    weighted_score = calculate_weighted_score(
        oof_07_auc, oof_07_ll, oof_90_auc, oof_90_ll, oof_120_auc, oof_120_ll,
    )

    return {
        "weighted_score": weighted_score,
        "oof_07_ll": oof_07_ll,
        "oof_07_auc": oof_07_auc,
        "oof_90_ll": oof_90_ll,
        "oof_90_auc": oof_90_auc,
        "oof_120_ll": oof_120_ll,
        "oof_120_auc": oof_120_auc,
        "monotonic_ok": monotonic_ok,
    }


def get_e1_oof(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run E1 harness and return OOF arrays (p7, p90, p120) and oof_mask. For E6 blending."""
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    y07 = train_df[TARGET_COLS[0]].values
    y90 = train_df[TARGET_COLS[1]].values
    y120 = train_df[TARGET_COLS[2]].values
    y1 = y07
    y2 = ((y90 == 1) & (y07 == 0)).astype(np.float64)
    y3 = ((y120 == 1) & (y90 == 0)).astype(np.float64)

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    oof_p7 = np.zeros(n_rows, dtype=float)
    oof_p90 = np.zeros(n_rows, dtype=float)
    oof_p120 = np.zeros(n_rows, dtype=float)
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

        for name, y_tr_vals, y_va_vals in [
            ("h1", y1[train_pos], y1[val_pos]),
            ("h2", y2[train_pos], y2[val_pos]),
            ("h3", y3[train_pos], y3[val_pos]),
        ]:
            train_set = lgb.Dataset(X_tr, label=y_tr_vals, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va_vals, free_raw_data=False)
            booster = lgb.train(
                params={**LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            h = booster.predict(X_va, num_iteration=booster.best_iteration)
            h = np.clip(h, *E0_CLIP)
            if name == "h1":
                h1_va = h
            elif name == "h2":
                h2_va = h
            else:
                h3_va = h

        p7_va = h1_va
        p90_va = 1.0 - (1.0 - h1_va) * (1.0 - h2_va)
        p120_va = 1.0 - (1.0 - h1_va) * (1.0 - h2_va) * (1.0 - h3_va)
        p7_va = np.clip(p7_va, *E0_CLIP)
        p90_va = np.clip(p90_va, *E0_CLIP)
        p120_va = np.clip(p120_va, *E0_CLIP)

        oof_p7[val_pos] = p7_va
        oof_p90[val_pos] = p90_va
        oof_p120[val_pos] = p120_va
        oof_mask[val_pos] = True

    return oof_p7, oof_p90, oof_p120, oof_mask


def main() -> None:
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E1 HAZARD DECOMPOSITION EXPERIMENT")
    logger.info("  Pass: ΔWeighted >= %.3f AND Δ7dLL <= %.3f", E1_DWEIGHTED_MIN, E1_D7D_LL_MAX)
    logger.info("  Guardrails: monotonicity; AUC must not collapse (tol=%.2f)", E1_AUC_COLLAPSE_TOL)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    # Baseline (E0)
    logger.info("\n--- E0 baseline (direct cumulative targets) ---")
    baseline = run_single_e0_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    logger.info("  Baseline weighted=%.6f  7d_ll=%.6f  7d_auc=%.6f  90d_auc=%.6f  120d_auc=%.6f",
                baseline["weighted_score"], baseline["oof_07_ll"],
                baseline["oof_07_auc"], baseline["oof_90_auc"], baseline["oof_120_auc"])

    # E1 (hazard decomposition)
    logger.info("\n--- E1 (interval hazards, reconstructed cumulative) ---")
    e1 = run_single_e1_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    logger.info("  E1     weighted=%.6f  7d_ll=%.6f  7d_auc=%.6f  90d_auc=%.6f  120d_auc=%.6f",
                e1["weighted_score"], e1["oof_07_ll"],
                e1["oof_07_auc"], e1["oof_90_auc"], e1["oof_120_auc"])

    # Deltas
    d_weighted = e1["weighted_score"] - baseline["weighted_score"]
    d_7d_ll = e1["oof_07_ll"] - baseline["oof_07_ll"]

    # Primary pass
    pass_weighted = d_weighted >= E1_DWEIGHTED_MIN
    pass_7d_ll = d_7d_ll <= E1_D7D_LL_MAX
    primary_pass = pass_weighted and pass_7d_ll

    # Guardrails: AUC must not collapse
    auc_07_ok = e1["oof_07_auc"] >= baseline["oof_07_auc"] - E1_AUC_COLLAPSE_TOL
    auc_90_ok = e1["oof_90_auc"] >= baseline["oof_90_auc"] - E1_AUC_COLLAPSE_TOL
    auc_120_ok = e1["oof_120_auc"] >= baseline["oof_120_auc"] - E1_AUC_COLLAPSE_TOL
    guardrail_auc = auc_07_ok and auc_90_ok and auc_120_ok
    guardrail_mono = e1["monotonic_ok"]

    logger.info("\n" + "=" * 70)
    logger.info("E1 RESULT")
    logger.info("  ΔWeighted = %.6f  (need >= %.3f)  %s", d_weighted, E1_DWEIGHTED_MIN, "OK" if pass_weighted else "FAIL")
    logger.info("  Δ7dLL     = %.6f  (need <= %.3f)  %s", d_7d_ll, E1_D7D_LL_MAX, "OK" if pass_7d_ll else "FAIL")
    logger.info("  Monotonicity: %s", "OK" if guardrail_mono else "FAIL")
    logger.info("  AUC no collapse (7d/90d/120d): %s", "OK" if guardrail_auc else "FAIL")
    if primary_pass and guardrail_auc and guardrail_mono:
        logger.info("  PASS: Adopt hazard decomposition.")
    else:
        logger.warning("  FAIL: Revert to cumulative direct targets.")
    logger.info("=" * 70)

    sys.exit(0 if (primary_pass and guardrail_auc and guardrail_mono) else 1)


if __name__ == "__main__":
    main()
