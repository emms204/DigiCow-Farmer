#!/usr/bin/env python3
"""
E7 Multi-cutoff model selection: rank by mean-minus-std over early/mid/late windows;
guardrail: winner must be top-2 in at least 3/4 windows.

Hypothesis E7: More robust cutoff coverage improves private LB transfer.
Experiment: Multi-cutoff model selection (4 time windows); rank candidates by mean(score) - std(score).
Primary metric: Final model choice criterion (mean - std).
Guardrail: No overfit to one window — winner must be top-2 in at least 3/4 windows.
Pass: Winner is top-2 in >= 3/4 windows.
If fail: Reject as unstable.

Usage:
    python e7_multi_cutoff.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.constants import DATE_COL, TARGET_COLS
from shared.data_loader import DataLoader
from shared.evaluation import calculate_weighted_score

from e0_reproducibility import (
    E0_MIN_TRAIN_SIZE,
    E0_N_SPLITS,
    E0_SEED,
    rolling_forward_splits,
)
from e6_oof_blend import (
    E6_WORST_FOLD_GUARDRAIL,
    _blend_oof,
    _weighted_score_from_predictions,
    _worst_fold_score,
    collect_oof_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

E7_TOP2_WINDOWS_MIN = 3  # winner must be rank 1 or 2 in at least this many windows
E7_N_WINDOWS = 4         # early, mid-early, mid-late, late


def _build_windows(splits: list[dict[str, Any]]) -> list[np.ndarray]:
    """Build 4 windows from 5 folds: w0=fold1, w1=fold2, w2=fold3, w3=fold4+fold5."""
    if len(splits) < 4:
        raise ValueError("Need at least 4 splits for 4 windows")
    return [
        splits[0]["val_pos"],
        splits[1]["val_pos"],
        splits[2]["val_pos"],
        np.concatenate([splits[3]["val_pos"], splits[4]["val_pos"]]),
    ]


def main() -> None:
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E7 MULTI-CUTOFF MODEL SELECTION")
    logger.info("  Criterion: mean(score) - std(score) over 4 windows")
    logger.info("  Pass: Winner is top-2 in >= %d/%d windows", E7_TOP2_WINDOWS_MIN, E7_N_WINDOWS)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    splits = rolling_forward_splits(
        train_df, n_splits=E0_N_SPLITS, min_train_size=E0_MIN_TRAIN_SIZE
    )
    windows = _build_windows(splits)
    window_names = ["early", "mid_early", "mid_late", "late"]

    # Collect OOF: E0..E5
    logger.info("\n--- Collecting OOF from E0–E5 ---")
    stack, oof_mask, names = collect_oof_models(train_df, prior_df)
    stack_07 = [s[0] for s in stack]
    stack_90 = [s[1] for s in stack]
    stack_120 = [s[2] for s in stack]
    K = len(names)

    # E6 blend: optimize weights (same as E6) and add as 7th model
    logger.info("  Computing E6 blend (constrained optimizer) ...")
    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]
    baseline_worst_fold = _worst_fold_score(
        stack_07[0], stack_90[0], stack_120[0], train_df, splits
    )

    def objective(w: np.ndarray) -> float:
        p07, p90, p120 = _blend_oof(w, stack_07, stack_90, stack_120)
        ws = _weighted_score_from_predictions(
            y_07, y_90, y_120,
            p07[oof_mask], p90[oof_mask], p120[oof_mask],
        )
        return -ws

    def constraint_worst_fold(w: np.ndarray) -> float:
        p07, p90, p120 = _blend_oof(w, stack_07, stack_90, stack_120)
        return _worst_fold_score(p07, p90, p120, train_df, splits) - baseline_worst_fold

    w0 = np.ones(K) / K
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if E6_WORST_FOLD_GUARDRAIL:
        constraints.append({"type": "ineq", "fun": constraint_worst_fold})

    res = minimize(
        objective, w0, method="SLSQP",
        bounds=[(0.0, 1.0)] * K,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    w_opt = np.clip(res.x, 0.0, 1.0)
    w_opt = w_opt / w_opt.sum()
    blend_07, blend_90, blend_120 = _blend_oof(w_opt, stack_07, stack_90, stack_120)

    # All 7 models: E0..E5 + E6_blend
    all_07 = stack_07 + [blend_07]
    all_90 = stack_90 + [blend_90]
    all_120 = stack_120 + [blend_120]
    all_names = names + ["E6_blend"]
    n_models = len(all_names)

    # Per-window weighted score: score[model][window]
    score_matrix = np.zeros((n_models, E7_N_WINDOWS))
    for w_idx, val_pos in enumerate(windows):
        y_07_w = train_df[TARGET_COLS[0]].iloc[val_pos].values
        y_90_w = train_df[TARGET_COLS[1]].iloc[val_pos].values
        y_120_w = train_df[TARGET_COLS[2]].iloc[val_pos].values
        for m in range(n_models):
            p07_w = all_07[m][val_pos]
            p90_w = all_90[m][val_pos]
            p120_w = all_120[m][val_pos]
            score_matrix[m, w_idx] = _weighted_score_from_predictions(
                y_07_w, y_90_w, y_120_w, p07_w, p90_w, p120_w
            )

    logger.info("\n--- Per-window weighted scores ---")
    for w_idx, wn in enumerate(window_names):
        logger.info("  %s: %s", wn, [round(score_matrix[m, w_idx], 4) for m in range(n_models)])

    # Criterion: mean - std (higher = more stable and good)
    means = score_matrix.mean(axis=1)
    stds = score_matrix.std(axis=1)
    # Avoid std=0 giving huge criterion; use std + 1e-6 or just mean - std
    criteria = means - stds
    winner_idx = int(np.argmax(criteria))
    winner_name = all_names[winner_idx]

    logger.info("\n--- Mean-minus-std ranking ---")
    order = np.argsort(criteria)[::-1]
    for i, m in enumerate(order):
        logger.info("  %d. %s  mean=%.6f  std=%.6f  mean-std=%.6f",
                    i + 1, all_names[m], means[m], stds[m], criteria[m])

    # Per-window ranks (1 = best)
    ranks = np.zeros((E7_N_WINDOWS, n_models), dtype=int)
    for w_idx in range(E7_N_WINDOWS):
        order_w = np.argsort(-score_matrix[:, w_idx])
        for rank, m in enumerate(order_w, start=1):
            ranks[w_idx, m] = rank

    winner_top2_count = sum(1 for w_idx in range(E7_N_WINDOWS) if ranks[w_idx, winner_idx] <= 2)
    pass_guardrail = winner_top2_count >= E7_TOP2_WINDOWS_MIN

    logger.info("\n--- Per-window ranks (1=best) ---")
    for w_idx, wn in enumerate(window_names):
        logger.info("  %s: %s", wn, dict(zip(all_names, ranks[w_idx].tolist())))

    logger.info("\n" + "=" * 70)
    logger.info("E7 RESULT")
    logger.info("  Winner (mean - std): %s  (mean=%.6f  std=%.6f  mean-std=%.6f)",
                winner_name, means[winner_idx], stds[winner_idx], criteria[winner_idx])
    logger.info("  Winner top-2 in %d/%d windows  (need >= %d)  %s",
                winner_top2_count, E7_N_WINDOWS, E7_TOP2_WINDOWS_MIN,
                "OK" if pass_guardrail else "FAIL")
    if pass_guardrail:
        logger.info("  PASS: Adopt %s as final model.", winner_name)
    else:
        logger.warning("  FAIL: Reject %s as unstable (not top-2 in enough windows).", winner_name)
    logger.info("=" * 70)

    sys.exit(0 if pass_guardrail else 1)


if __name__ == "__main__":
    main()
