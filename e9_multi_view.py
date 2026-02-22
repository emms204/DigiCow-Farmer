#!/usr/bin/env python3
"""
E9 Plan 9 (Multi-View Ensemble) vs E6 winner.

Decision rule: Promote Plan 9 only if all hold vs E6:
  1. ΔWeighted >= +0.008
  2. Δ7dLL <= -0.002
  3. worst-fold >= E6 worst-fold
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.constants import DATE_COL, TARGET_COLS
from shared.data_loader import DataLoader

from e0_reproducibility import E0_MIN_TRAIN_SIZE, E0_N_SPLITS, E0_SEED, rolling_forward_splits
from e6_oof_blend import (
    E6_WORST_FOLD_GUARDRAIL,
    _blend_oof,
    _weighted_score_from_predictions,
    _worst_fold_score,
    collect_oof_models,
)
from plan9.model import run_plan9_oof

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

E9_WEIGHTED_DELTA_MIN = 0.008
E9_7D_LL_DELTA_MAX = -0.002


def main() -> None:
    logger.info("=" * 70)
    logger.info("E9 PLAN 9 (MULTI-VIEW) vs E6 WINNER")
    logger.info("  Pass: ΔWeighted >= +%.3f, Δ7dLL <= %.3f, worst-fold >= E6", E9_WEIGHTED_DELTA_MIN, E9_7D_LL_DELTA_MAX)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    splits = rolling_forward_splits(train_df, n_splits=E0_N_SPLITS, min_train_size=E0_MIN_TRAIN_SIZE)

    # E6 baseline
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"
    logger.info("\n--- E6 baseline ---")
    stack, oof_mask, names = collect_oof_models(train_df, prior_df)
    stack_07 = [s[0] for s in stack]
    stack_90 = [s[1] for s in stack]
    stack_120 = [s[2] for s in stack]
    K = len(names)
    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]
    baseline_worst = _worst_fold_score(stack_07[0], stack_90[0], stack_120[0], train_df, splits)

    def objective(w):
        p07, p90, p120 = _blend_oof(w, stack_07, stack_90, stack_120)
        return -_weighted_score_from_predictions(y_07, y_90, y_120, p07[oof_mask], p90[oof_mask], p120[oof_mask])

    def constraint_worst(w):
        p07, p90, p120 = _blend_oof(w, stack_07, stack_90, stack_120)
        return _worst_fold_score(p07, p90, p120, train_df, splits) - baseline_worst

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if E6_WORST_FOLD_GUARDRAIL:
        constraints.append({"type": "ineq", "fun": constraint_worst})
    res = minimize(objective, np.ones(K) / K, method="SLSQP", bounds=[(0.0, 1.0)] * K, constraints=constraints, options={"maxiter": 200, "ftol": 1e-9})
    w_opt = np.clip(res.x, 0.0, 1.0) / np.clip(res.x, 0.0, 1.0).sum()
    e6_07, e6_90, e6_120 = _blend_oof(w_opt, stack_07, stack_90, stack_120)
    e6_weighted = _weighted_score_from_predictions(y_07, y_90, y_120, e6_07[oof_mask], e6_90[oof_mask], e6_120[oof_mask])
    e6_7d_ll = log_loss(y_07, e6_07[oof_mask])
    e6_worst_fold = _worst_fold_score(e6_07, e6_90, e6_120, train_df, splits)
    logger.info("  E6 weighted=%.6f  7d_ll=%.6f  worst_fold=%.6f", e6_weighted, e6_7d_ll, e6_worst_fold)

    # Plan 9
    logger.info("\n--- Plan 9 (multi-view ensemble) ---")
    p9_07, p9_90, p9_120, p9_mask = run_plan9_oof(train_df, prior_df, splits=splits, seed=E0_SEED, baseline_worst_fold=e6_worst_fold)
    y_07_p9 = train_df[TARGET_COLS[0]].values[p9_mask]
    y_90_p9 = train_df[TARGET_COLS[1]].values[p9_mask]
    y_120_p9 = train_df[TARGET_COLS[2]].values[p9_mask]
    p9_weighted = _weighted_score_from_predictions(y_07_p9, y_90_p9, y_120_p9, p9_07[p9_mask], p9_90[p9_mask], p9_120[p9_mask])
    p9_7d_ll = log_loss(y_07_p9, p9_07[p9_mask])
    p9_worst_fold = _worst_fold_score(p9_07, p9_90, p9_120, train_df, splits)
    logger.info("  Plan9 weighted=%.6f  7d_ll=%.6f  worst_fold=%.6f", p9_weighted, p9_7d_ll, p9_worst_fold)

    d_weighted = p9_weighted - e6_weighted
    d_7d_ll = p9_7d_ll - e6_7d_ll
    pass_weighted = d_weighted >= E9_WEIGHTED_DELTA_MIN
    pass_7d_ll = d_7d_ll <= E9_7D_LL_DELTA_MAX
    pass_worst = p9_worst_fold >= e6_worst_fold
    primary_pass = pass_weighted and pass_7d_ll and pass_worst

    logger.info("\n" + "=" * 70)
    logger.info("E9 RESULT")
    logger.info("  ΔWeighted   = %.6f  (need >= +%.3f)  %s", d_weighted, E9_WEIGHTED_DELTA_MIN, "OK" if pass_weighted else "FAIL")
    logger.info("  Δ7dLL       = %.6f  (need <= %.3f)  %s", d_7d_ll, E9_7D_LL_DELTA_MAX, "OK" if pass_7d_ll else "FAIL")
    logger.info("  Worst-fold  = %.6f  (need >= %.6f)  %s", p9_worst_fold, e6_worst_fold, "OK" if pass_worst else "FAIL")
    if primary_pass:
        logger.info("  PASS: Promote Plan 9.")
    else:
        logger.warning("  FAIL: Keep E6 winner.")
    logger.info("=" * 70)
    sys.exit(0 if primary_pass else 1)


if __name__ == "__main__":
    main()
