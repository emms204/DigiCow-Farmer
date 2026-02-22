#!/usr/bin/env python3
"""
E0 Reproducibility experiment: locked evaluation harness for Plan 1 (LGBM) + minimal features.

Hypothesis E0: Validation mismatch is causing false gains.
Experiment: Lock one evaluation harness (non-overlap forward folds, as-of features per fold,
            same seeds, same clip/calibration policy) and run multiple times.
Primary pass metric: Reproducible baseline score (weighted competition score).
Guardrails: Fold variance tracked; latest fold reported separately.
Pass threshold: Same config rerun changes < 0.002 weighted.
If fail: Stop all feature/model work until fixed.

Usage:
    python e0_reproducibility.py              # Run 3 times, report pass/fail
    python e0_reproducibility.py --runs 5   # Run 5 times
    python e0_reproducibility.py --dry-run   # Single run, no pass/fail
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Locked E0 config (do not change without documenting) ─────────────────────
E0_N_SPLITS = 5
E0_MIN_TRAIN_SIZE = 0.3
E0_CLIP = (PROB_CLIP_MIN, PROB_CLIP_MAX)
E0_SEED = RANDOM_SEED  # 42


def rolling_forward_splits(
    df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    min_train_size: float = E0_MIN_TRAIN_SIZE,
) -> list[dict[str, Any]]:
    """Non-overlapping forward-time splits. Train = past, val = next window."""
    if DATE_COL not in df.columns:
        raise ValueError(f"{DATE_COL} missing")

    ordered = df[[DATE_COL]].copy()
    ordered["_pos"] = np.arange(len(df), dtype=int)
    ordered = ordered.sort_values(DATE_COL)

    min_date = ordered[DATE_COL].min()
    max_date = ordered[DATE_COL].max()

    cutoff_fracs = [
        min_train_size + (i + 1) * (1 - min_train_size) / (n_splits + 1)
        for i in range(n_splits + 1)
    ]
    cutoffs = [min_date + (max_date - min_date) * frac for frac in cutoff_fracs]

    splits = []
    for i in range(n_splits):
        start, end = cutoffs[i], cutoffs[i + 1]
        train_mask = ordered[DATE_COL] < start
        val_mask = (ordered[DATE_COL] >= start) & (ordered[DATE_COL] < end)
        train_pos = ordered.loc[train_mask, "_pos"].to_numpy(dtype=int)
        val_pos = ordered.loc[val_mask, "_pos"].to_numpy(dtype=int)
        if len(train_pos) == 0 or len(val_pos) == 0:
            continue
        splits.append({
            "fold": i + 1,
            "train_pos": train_pos,
            "val_pos": val_pos,
            "train_cutoff": start,
            "val_end": end,
        })
    if not splits:
        raise ValueError("No valid forward splits")
    return splits


def run_single_e0_harness(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E0_N_SPLITS,
    seed: int = E0_SEED,
) -> dict[str, Any]:
    """
    One run of the locked E0 harness: Plan 1 (LGBM) + minimal features,
    non-overlapping forward folds, as-of features per fold, fixed seeds, clip.
    Returns weighted_score, per_fold_weighted_scores, fold_std, last_fold_score.
    """
    import lightgbm as lgb
    from plan1.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, NUM_BOOST_ROUND
    from sklearn.metrics import log_loss, roc_auc_score

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
    fold_metrics_list: list[dict[str, float]] = []

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

        fold_metrics: dict[str, float] = {}
        for target in TARGET_COLS:
            y = train_df[target]
            y_tr = y.iloc[train_pos].values
            y_va = y.iloc[val_pos].values

            train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            val_set = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
            callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
            booster = lgb.train(
                params={**LGBM_PARAMS, "random_state": seed},
                train_set=train_set,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[train_set, val_set],
                valid_names=["train", "valid"],
                callbacks=callbacks,
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
        fold_metrics_list.append(fold_metrics)
        logger.info(
            "  Fold %d: train=%d val=%d weighted=%.6f",
            fold, len(train_pos), len(val_pos), fold_weighted,
        )

    # OOF competition score (primary metric)
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
    fold_std = float(np.std(per_fold_weighted_scores)) if len(per_fold_weighted_scores) > 1 else 0.0
    last_fold_score = per_fold_weighted_scores[-1] if per_fold_weighted_scores else 0.0

    return {
        "weighted_score": weighted_score,
        "per_fold_weighted_scores": per_fold_weighted_scores,
        "fold_std": fold_std,
        "last_fold_score": last_fold_score,
        "oof_07_ll": oof_07_ll,
        "oof_07_auc": oof_07_auc,
        "oof_90_ll": oof_90_ll,
        "oof_90_auc": oof_90_auc,
        "oof_120_ll": oof_120_ll,
        "oof_120_auc": oof_120_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E0 reproducibility: Plan 1 + minimal features")
    parser.add_argument("--runs", type=int, default=3, help="Number of identical runs to compare")
    parser.add_argument("--threshold", type=float, default=0.002, help="Max allowed score delta for pass")
    parser.add_argument("--dry-run", action="store_true", help="Single run only, no pass/fail")
    parser.add_argument("--n-splits", type=int, default=E0_N_SPLITS, help="Forward CV folds (locked default 5)")
    args = parser.parse_args()

    # Force minimal features for this experiment (do not rely on env from caller)
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E0 REPRODUCIBILITY EXPERIMENT")
    logger.info("  Config: Plan 1 (LGBM) + minimal features, forward folds, as-of features")
    logger.info("  Seed=%s, n_splits=%s, clip=%s", E0_SEED, args.n_splits, E0_CLIP)
    logger.info("  Pass: max(run scores) - min(run scores) < %s", args.threshold)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    if args.dry_run:
        result = run_single_e0_harness(train_df, prior_df, n_splits=args.n_splits, seed=E0_SEED)
        logger.info("  Weighted score (primary): %.6f", result["weighted_score"])
        logger.info("  7d  LogLoss=%.6f  AUC=%.6f", result["oof_07_ll"], result["oof_07_auc"])
        logger.info("  90d LogLoss=%.6f  AUC=%.6f", result["oof_90_ll"], result["oof_90_auc"])
        logger.info("  120d LogLoss=%.6f AUC=%.6f", result["oof_120_ll"], result["oof_120_auc"])
        logger.info("  Fold std (guardrail):      %.6f", result["fold_std"])
        logger.info("  Last fold score:           %.6f", result["last_fold_score"])
        return

    scores: list[float] = []
    for r in range(args.runs):
        logger.info("\n--- Run %d / %d ---", r + 1, args.runs)
        result = run_single_e0_harness(train_df, prior_df, n_splits=args.n_splits, seed=E0_SEED)
        scores.append(result["weighted_score"])
        logger.info(
            "  Run %d weighted=%.6f 7d_ll=%.6f 7d_auc=%.6f (fold_std=%.6f)",
            r + 1, result["weighted_score"], result["oof_07_ll"], result["oof_07_auc"], result["fold_std"],
        )

    max_delta = max(scores) - min(scores)
    passed = max_delta < args.threshold

    logger.info("\n" + "=" * 70)
    logger.info("E0 RESULT")
    logger.info("  Scores: %s", [round(s, 6) for s in scores])
    logger.info("  Max delta: %.6f", max_delta)
    logger.info("  Threshold: %.6f", args.threshold)
    if passed:
        logger.info("  PASS: Reproducible baseline (delta < threshold)")
        logger.info("  Baseline score (mean): %.6f", float(np.mean(scores)))
    else:
        logger.warning("  FAIL: Score changed by >= %.6f across runs. Stop feature/model work until fixed.", args.threshold)
    logger.info("=" * 70)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
