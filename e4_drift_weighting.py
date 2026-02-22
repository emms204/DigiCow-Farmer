#!/usr/bin/env python3
"""
E4 Drift-aware weighting: recency + adversarial similarity weighting on train rows.

Hypothesis E4: Drift-aware weighting improves transfer to latest period.
Experiment: Recency weighting + adversarial similarity weighting on train rows.
Primary pass metric: Latest-fold weighted score.
Guardrails: Overall weighted must not drop.
Pass threshold: Latest fold +0.01 and overall >= +0.003.
If fail: Disable weighting.

Usage:
    python e4_drift_weighting.py
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
from sklearn.linear_model import LogisticRegression
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

E4_LATEST_FOLD_DELTA_MIN = 0.01   # latest-fold weighted score must improve by at least this
E4_OVERALL_DELTA_MIN = 0.003      # overall weighted score must improve by at least this
E4_RECENCY_ALPHA = 1.0            # recency weight = 1 + alpha * (recency_frac)
E4_ADV_STRENGTH = 1.0              # adversarial weight = 1 + strength * P(val|x) for train


def run_e4_weighted_harness(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> dict[str, Any]:
    """Run E0-style harness but with recency + adversarial sample weights per fold."""
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    oof_preds = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
    oof_mask = np.zeros(n_rows, dtype=bool)
    per_fold_weighted_scores: list[float] = []

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

        # Recency weights: higher weight for more recent train rows
        train_dates = pd.to_datetime(train_df[DATE_COL].iloc[train_pos])
        min_d = train_dates.min()
        max_d = train_dates.max()
        delta_days = (train_dates - min_d).dt.total_seconds() / 86400.0
        span = (max_d - min_d).total_seconds() / 86400.0 if max_d > min_d else 1.0
        recency_frac = delta_days / span
        recency_w = 1.0 + E4_RECENCY_ALPHA * recency_frac
        recency_w = recency_w / recency_w.mean()

        # Adversarial similarity: train discriminator train=0 vs val=1
        n_tr, n_va = len(train_pos), len(val_pos)
        X_adv = np.vstack([X_tr, X_va])
        y_adv = np.array([0] * n_tr + [1] * n_va)
        disc = LogisticRegression(max_iter=500, random_state=seed)
        disc.fit(X_adv, y_adv)
        p_val_train = disc.predict_proba(X_tr)[:, 1]
        adv_w = 1.0 + E4_ADV_STRENGTH * p_val_train
        adv_w = adv_w / adv_w.mean()

        # Combine and normalize so sum = n_train (common for LGBM)
        w_tr = recency_w * adv_w
        w_tr = w_tr / w_tr.sum() * len(w_tr)

        fold_metrics: dict[str, float] = {}
        for target in TARGET_COLS:
            y_tr = train_df[target].iloc[train_pos].values
            y_va = train_df[target].iloc[val_pos].values

            train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            booster = lgb.train(
                params={**LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_va = booster.predict(X_va, num_iteration=booster.best_iteration)
            pred_va = np.clip(pred_va, *E0_CLIP)

            oof_preds[target][val_pos] = pred_va
            oof_mask[val_pos] = True

            ll = log_loss(y_va, pred_va)
            auc = roc_auc_score(y_va, pred_va) if len(np.unique(y_va)) > 1 else 0.5
            fold_metrics[f"{target}_ll"] = ll
            fold_metrics[f"{target}_auc"] = auc

        fold_weighted = calculate_weighted_score(
            fold_metrics["adopted_within_07_days_auc"],
            fold_metrics["adopted_within_07_days_ll"],
            fold_metrics["adopted_within_90_days_auc"],
            fold_metrics["adopted_within_90_days_ll"],
            fold_metrics["adopted_within_120_days_auc"],
            fold_metrics["adopted_within_120_days_ll"],
        )
        per_fold_weighted_scores.append(fold_weighted)
        logger.info(
            "  Fold %d: train=%d val=%d weighted=%.6f (weighted train)",
            fold, len(train_pos), len(val_pos), fold_weighted,
        )

    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]
    p_07 = oof_preds[TARGET_COLS[0]][oof_mask]
    p_90 = oof_preds[TARGET_COLS[1]][oof_mask]
    p_120 = oof_preds[TARGET_COLS[2]][oof_mask]

    oof_07_ll = log_loss(y_07, p_07)
    oof_07_auc = roc_auc_score(y_07, p_07) if len(np.unique(y_07)) > 1 else 0.5
    oof_90_ll = log_loss(y_90, p_90)
    oof_90_auc = roc_auc_score(y_90, p_90) if len(np.unique(y_90)) > 1 else 0.5
    oof_120_ll = log_loss(y_120, p_120)
    oof_120_auc = roc_auc_score(y_120, p_120) if len(np.unique(y_120)) > 1 else 0.5

    weighted_score = calculate_weighted_score(
        oof_07_auc, oof_07_ll, oof_90_auc, oof_90_ll, oof_120_auc, oof_120_ll,
    )
    latest_fold_score = per_fold_weighted_scores[-1] if per_fold_weighted_scores else 0.0

    return {
        "weighted_score": weighted_score,
        "per_fold_weighted_scores": per_fold_weighted_scores,
        "latest_fold_score": latest_fold_score,
    }


def get_e4_oof(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run E4 weighted harness and return OOF dict (target -> array) and oof_mask. For E6 blending."""
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND

    np.random.seed(seed)
    train_df = train_df.copy()
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    splits = rolling_forward_splits(train_df, n_splits=n_splits, min_train_size=E0_MIN_TRAIN_SIZE)
    n_rows = len(train_df)
    oof_preds = {t: np.zeros(n_rows, dtype=float) for t in TARGET_COLS}
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

        train_dates = pd.to_datetime(train_df[DATE_COL].iloc[train_pos])
        min_d = train_dates.min()
        max_d = train_dates.max()
        delta_days = (train_dates - min_d).dt.total_seconds() / 86400.0
        span = (max_d - min_d).total_seconds() / 86400.0 if max_d > min_d else 1.0
        recency_frac = delta_days / span
        recency_w = 1.0 + E4_RECENCY_ALPHA * recency_frac
        recency_w = recency_w / recency_w.mean()

        n_tr, n_va = len(train_pos), len(val_pos)
        X_adv = np.vstack([X_tr, X_va])
        y_adv = np.array([0] * n_tr + [1] * n_va)
        disc = LogisticRegression(max_iter=500, random_state=seed)
        disc.fit(X_adv, y_adv)
        p_val_train = disc.predict_proba(X_tr)[:, 1]
        adv_w = 1.0 + E4_ADV_STRENGTH * p_val_train
        adv_w = adv_w / adv_w.mean()

        w_tr = recency_w * adv_w
        w_tr = w_tr / w_tr.sum() * len(w_tr)

        for target in TARGET_COLS:
            y_tr = train_df[target].iloc[train_pos].values
            train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=train_df[target].iloc[val_pos].values, free_raw_data=False)
            booster = lgb.train(
                params={**LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred_va = booster.predict(X_va, num_iteration=booster.best_iteration)
            oof_preds[target][val_pos] = np.clip(pred_va, *E0_CLIP)
            oof_mask[val_pos] = True

    return oof_preds, oof_mask


def main() -> None:
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E4 DRIFT-AWARE WEIGHTING EXPERIMENT")
    logger.info("  Pass: latest_fold +%.2f and overall >= +%.3f", E4_LATEST_FOLD_DELTA_MIN, E4_OVERALL_DELTA_MIN)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    logger.info("\n--- E0 baseline (no weighting) ---")
    baseline = run_single_e0_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    baseline_latest = baseline["last_fold_score"]
    baseline_overall = baseline["weighted_score"]
    logger.info("  Baseline latest_fold=%.6f  overall=%.6f", baseline_latest, baseline_overall)

    logger.info("\n--- E4 (recency + adversarial weighting) ---")
    e4 = run_e4_weighted_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    e4_latest = e4["latest_fold_score"]
    e4_overall = e4["weighted_score"]
    logger.info("  E4     latest_fold=%.6f  overall=%.6f", e4_latest, e4_overall)

    d_latest = e4_latest - baseline_latest
    d_overall = e4_overall - baseline_overall
    pass_latest = d_latest >= E4_LATEST_FOLD_DELTA_MIN
    pass_overall = d_overall >= E4_OVERALL_DELTA_MIN
    primary_pass = pass_latest and pass_overall

    logger.info("\n" + "=" * 70)
    logger.info("E4 RESULT")
    logger.info("  Δlatest_fold = %.6f  (need >= +%.2f)  %s", d_latest, E4_LATEST_FOLD_DELTA_MIN, "OK" if pass_latest else "FAIL")
    logger.info("  Δoverall     = %.6f  (need >= +%.3f)  %s", d_overall, E4_OVERALL_DELTA_MIN, "OK" if pass_overall else "FAIL")
    if primary_pass:
        logger.info("  PASS: Adopt drift-aware weighting.")
    else:
        logger.warning("  FAIL: Disable weighting.")
    logger.info("=" * 70)

    sys.exit(0 if primary_pass else 1)


if __name__ == "__main__":
    main()
