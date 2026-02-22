#!/usr/bin/env python3
"""
E6 OOF constrained blending: blend top models from E0–E5 via constrained optimizer on forward OOF.

Hypothesis E6: OOF constrained blending across families/seeds reduces variance.
Experiment: Blend OOF from E0, E1, E2, E3 (isotonic), E4, E5; optimize weights (w >= 0, sum=1)
            to maximize weighted score, with constraint worst-fold >= baseline worst-fold.
Primary pass metric: Weighted score.
Guardrail: Worst-fold score not worse than baseline.
Pass: ΔWeighted >= +0.006 and blend worst-fold >= baseline worst-fold.
If fail: Keep best single model.

Usage:
    python e6_oof_blend.py
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
    run_single_e0_harness,
)
from e1_hazard_decomposition import get_e1_oof
from e2_dual_auc_ll import get_e2_oof
from e3_calibration_blend import apply_calibrators, get_raw_oof
from e4_drift_weighting import get_e4_oof
from e5_safe_encodings import get_e5_oof

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

E6_WEIGHTED_DELTA_MIN = 0.006   # blend must improve weighted score by at least this
E6_WORST_FOLD_GUARDRAIL = True # blend worst-fold must be >= baseline worst-fold


def _weighted_score_from_predictions(
    y_07: np.ndarray, y_90: np.ndarray, y_120: np.ndarray,
    p_07: np.ndarray, p_90: np.ndarray, p_120: np.ndarray,
) -> float:
    """Compute competition weighted score from arrays of same length."""
    ll_07 = log_loss(y_07, p_07)
    ll_90 = log_loss(y_90, p_90)
    ll_120 = log_loss(y_120, p_120)
    auc_07 = roc_auc_score(y_07, p_07) if len(np.unique(y_07)) > 1 else 0.5
    auc_90 = roc_auc_score(y_90, p_90) if len(np.unique(y_90)) > 1 else 0.5
    auc_120 = roc_auc_score(y_120, p_120) if len(np.unique(y_120)) > 1 else 0.5
    return calculate_weighted_score(auc_07, ll_07, auc_90, ll_90, auc_120, ll_120)


def _blend_oof(
    w: np.ndarray,
    stack_07: list[np.ndarray],
    stack_90: list[np.ndarray],
    stack_120: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend OOF: blend_07 = sum_k w[k] * stack_07[k], etc. Weights normalized to sum=1."""
    w = np.clip(w, 0.0, 1.0)
    w = w / w.sum()
    n = len(stack_07[0])
    p07 = np.zeros(n)
    p90 = np.zeros(n)
    p120 = np.zeros(n)
    for k in range(len(w)):
        p07 += w[k] * stack_07[k]
        p90 += w[k] * stack_90[k]
        p120 += w[k] * stack_120[k]
    return p07, p90, p120


def _worst_fold_score(
    p_07: np.ndarray, p_90: np.ndarray, p_120: np.ndarray,
    train_df: pd.DataFrame,
    splits: list[dict[str, Any]],
) -> float:
    """Return minimum over folds of fold-level weighted score."""
    scores = []
    for split in splits:
        val_pos = split["val_pos"]
        y_07 = train_df[TARGET_COLS[0]].iloc[val_pos].values
        y_90 = train_df[TARGET_COLS[1]].iloc[val_pos].values
        y_120 = train_df[TARGET_COLS[2]].iloc[val_pos].values
        p07_f = p_07[val_pos]
        p90_f = p_90[val_pos]
        p120_f = p_120[val_pos]
        scores.append(_weighted_score_from_predictions(y_07, y_90, y_120, p07_f, p90_f, p120_f))
    return float(min(scores))


def collect_oof_models(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, list[str]]:
    """Collect (p07, p90, p120) and oof_mask from E0, E1, E2, E3, E4, E5. Returns (stack, oof_mask, names)."""
    stack_07: list[np.ndarray] = []
    stack_90: list[np.ndarray] = []
    stack_120: list[np.ndarray] = []
    names = ["E0", "E1", "E2", "E3_isotonic", "E4", "E5"]
    oof_mask_ref: np.ndarray | None = None

    # E0: raw OOF from E3 helper
    logger.info("  Collecting E0 (raw LGBM) OOF ...")
    oof_preds, oof_mask = get_raw_oof(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    oof_mask_ref = oof_mask
    stack_07.append(oof_preds[TARGET_COLS[0]])
    stack_90.append(oof_preds[TARGET_COLS[1]])
    stack_120.append(oof_preds[TARGET_COLS[2]])

    # E1: hazard
    logger.info("  Collecting E1 (hazard) OOF ...")
    p7, p90, p120, mask = get_e1_oof(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    assert np.array_equal(mask, oof_mask_ref), "E1 oof_mask mismatch"
    stack_07.append(p7)
    stack_90.append(p90)
    stack_120.append(p120)

    # E2: dual LL
    logger.info("  Collecting E2 (dual LL) OOF ...")
    p07, p90, p120, mask = get_e2_oof(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    assert np.array_equal(mask, oof_mask_ref), "E2 oof_mask mismatch"
    stack_07.append(p07)
    stack_90.append(p90)
    stack_120.append(p120)

    # E3: isotonic (raw OOF already loaded; apply calibrators)
    logger.info("  Collecting E3 (isotonic) OOF ...")
    _, calibrated = apply_calibrators(oof_preds, oof_mask, train_df)
    stack_07.append(calibrated[TARGET_COLS[0]]["isotonic"])
    stack_90.append(calibrated[TARGET_COLS[1]]["isotonic"])
    stack_120.append(calibrated[TARGET_COLS[2]]["isotonic"])

    # E4: drift weighting
    logger.info("  Collecting E4 (drift weighting) OOF ...")
    oof_preds_e4, mask_e4 = get_e4_oof(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    assert np.array_equal(mask_e4, oof_mask_ref), "E4 oof_mask mismatch"
    stack_07.append(oof_preds_e4[TARGET_COLS[0]])
    stack_90.append(oof_preds_e4[TARGET_COLS[1]])
    stack_120.append(oof_preds_e4[TARGET_COLS[2]])

    # E5: encodings
    logger.info("  Collecting E5 (encodings) OOF ...")
    oof_preds_e5, mask_e5 = get_e5_oof(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    assert np.array_equal(mask_e5, oof_mask_ref), "E5 oof_mask mismatch"
    stack_07.append(oof_preds_e5[TARGET_COLS[0]])
    stack_90.append(oof_preds_e5[TARGET_COLS[1]])
    stack_120.append(oof_preds_e5[TARGET_COLS[2]])

    stack = list(zip(stack_07, stack_90, stack_120))
    return stack, oof_mask_ref, names


def main() -> None:
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    logger.info("=" * 70)
    logger.info("E6 OOF CONSTRAINED BLENDING")
    logger.info("  Pass: ΔWeighted >= +%.3f and worst-fold >= baseline worst-fold", E6_WEIGHTED_DELTA_MIN)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    splits = rolling_forward_splits(
        train_df, n_splits=E0_N_SPLITS, min_train_size=E0_MIN_TRAIN_SIZE
    )

    # Baseline (E0) for weighted score and worst-fold
    logger.info("\n--- E0 baseline ---")
    baseline = run_single_e0_harness(train_df, prior_df, n_splits=E0_N_SPLITS, seed=E0_SEED)
    baseline_weighted = baseline["weighted_score"]
    baseline_worst_fold = min(baseline["per_fold_weighted_scores"])
    logger.info("  Baseline weighted=%.6f  worst_fold=%.6f", baseline_weighted, baseline_worst_fold)

    # Collect OOF from all models
    logger.info("\n--- Collecting OOF from E0–E5 ---")
    stack, oof_mask, names = collect_oof_models(train_df, prior_df)
    K = len(stack)
    stack_07 = [s[0] for s in stack]
    stack_90 = [s[1] for s in stack]
    stack_120 = [s[2] for s in stack]

    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]

    # Per-model OOF scores (for logging)
    for k, name in enumerate(names):
        ws = _weighted_score_from_predictions(
            y_07, y_90, y_120,
            stack_07[k][oof_mask], stack_90[k][oof_mask], stack_120[k][oof_mask],
        )
        logger.info("  %s OOF weighted=%.6f", name, ws)

    # Optimize blend: max weighted score, s.t. w >= 0, sum(w)=1, worst_fold >= baseline_worst_fold
    def objective(w: np.ndarray) -> float:
        p07, p90, p120 = _blend_oof(w, stack_07, stack_90, stack_120)
        ws = _weighted_score_from_predictions(
            y_07, y_90, y_120,
            p07[oof_mask], p90[oof_mask], p120[oof_mask],
        )
        return -ws

    def constraint_worst_fold(w: np.ndarray) -> float:
        """Inequality g(w) >= 0: worst_fold(blend) - baseline_worst_fold >= 0."""
        p07, p90, p120 = _blend_oof(w, stack_07, stack_90, stack_120)
        wf = _worst_fold_score(p07, p90, p120, train_df, splits)
        return wf - baseline_worst_fold

    w0 = np.ones(K) / K
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if E6_WORST_FOLD_GUARDRAIL:
        constraints.append({"type": "ineq", "fun": constraint_worst_fold})

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * K,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    w_opt = np.clip(res.x, 0.0, 1.0)
    w_opt = w_opt / w_opt.sum()

    blend_07, blend_90, blend_120 = _blend_oof(w_opt, stack_07, stack_90, stack_120)
    blend_weighted = _weighted_score_from_predictions(
        y_07, y_90, y_120,
        blend_07[oof_mask], blend_90[oof_mask], blend_120[oof_mask],
    )
    blend_worst_fold = _worst_fold_score(blend_07, blend_90, blend_120, train_df, splits)

    logger.info("\n--- E6 optimized blend ---")
    logger.info("  Weights: %s", dict(zip(names, [round(float(w), 4) for w in w_opt])))
    logger.info("  Blend weighted=%.6f  worst_fold=%.6f", blend_weighted, blend_worst_fold)

    d_weighted = blend_weighted - baseline_weighted
    pass_weighted = d_weighted >= E6_WEIGHTED_DELTA_MIN
    pass_worst_fold = blend_worst_fold >= baseline_worst_fold
    primary_pass = pass_weighted and pass_worst_fold

    logger.info("\n" + "=" * 70)
    logger.info("E6 RESULT")
    logger.info("  ΔWeighted    = %.6f  (need >= +%.3f)  %s", d_weighted, E6_WEIGHTED_DELTA_MIN, "OK" if pass_weighted else "FAIL")
    logger.info("  Worst-fold   = %.6f  (need >= %.6f)  %s", blend_worst_fold, baseline_worst_fold, "OK" if pass_worst_fold else "FAIL")
    if primary_pass:
        logger.info("  PASS: Adopt OOF blend.")
    else:
        logger.warning("  FAIL: Keep best single model.")
    logger.info("=" * 70)

    sys.exit(0 if primary_pass else 1)


if __name__ == "__main__":
    main()
