#!/usr/bin/env python3
"""
E8 Plan 8 (Cohort-Aware Mixture + Multi-Calibration) vs E6 winner.

Same locked E0 harness: forward folds, as-of features.
Pass: ΔWeighted >= +0.006 vs E6, Δ7dLL <= -0.002, worst-fold >= E6 worst-fold.
If fail: Likely ceiling with current data/feature universe.
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
from plan8.model import run_plan8_oof

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

E8_WEIGHTED_DELTA_MIN = 0.006   # vs E6 winner
E8_7D_LL_DELTA_MAX = -0.002     # 7d LL must improve by at least 0.002


def main() -> None:
    logger.info("=" * 70)
    logger.info("E8 PLAN 8 (COHORT MIXTURE) vs E6 WINNER")
    logger.info("  Pass: ΔWeighted >= +%.3f, Δ7dLL <= %.3f, worst-fold >= E6", E8_WEIGHTED_DELTA_MIN, E8_7D_LL_DELTA_MAX)
    logger.info("=" * 70)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    splits = rolling_forward_splits(
        train_df, n_splits=E0_N_SPLITS, min_train_size=E0_MIN_TRAIN_SIZE
    )

    # E6 baseline: same locked setup (minimal features) for comparable baseline
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"
    logger.info("\n--- E6 baseline (blend) ---")
    stack, oof_mask, names = collect_oof_models(train_df, prior_df)
    stack_07 = [s[0] for s in stack]
    stack_90 = [s[1] for s in stack]
    stack_120 = [s[2] for s in stack]
    K = len(names)

    y_07 = train_df[TARGET_COLS[0]].values[oof_mask]
    y_90 = train_df[TARGET_COLS[1]].values[oof_mask]
    y_120 = train_df[TARGET_COLS[2]].values[oof_mask]
    baseline_worst_fold = _worst_fold_score(stack_07[0], stack_90[0], stack_120[0], train_df, splits)

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
    e6_07, e6_90, e6_120 = _blend_oof(w_opt, stack_07, stack_90, stack_120)

    e6_weighted = _weighted_score_from_predictions(
        y_07, y_90, y_120,
        e6_07[oof_mask], e6_90[oof_mask], e6_120[oof_mask],
    )
    e6_7d_ll = log_loss(y_07, e6_07[oof_mask])
    e6_worst_fold = _worst_fold_score(e6_07, e6_90, e6_120, train_df, splits)
    logger.info("  E6 weighted=%.6f  7d_ll=%.6f  worst_fold=%.6f", e6_weighted, e6_7d_ll, e6_worst_fold)

    # Plan 8 OOF (uses warm/cold feature groups; do not use minimal)
    os.environ["DIGICOW_MINIMAL_FEATURES"] = "0"
    logger.info("\n--- Plan 8 (cohort mixture + multi-calibration) ---")
    p8_07, p8_90, p8_120, p8_mask = run_plan8_oof(train_df, prior_df, splits, seed=E0_SEED)

    y_07_p8 = train_df[TARGET_COLS[0]].values[p8_mask]
    y_90_p8 = train_df[TARGET_COLS[1]].values[p8_mask]
    y_120_p8 = train_df[TARGET_COLS[2]].values[p8_mask]
    p8_weighted = _weighted_score_from_predictions(
        y_07_p8, y_90_p8, y_120_p8,
        p8_07[p8_mask], p8_90[p8_mask], p8_120[p8_mask],
    )
    p8_7d_ll = log_loss(y_07_p8, p8_07[p8_mask])
    p8_worst_fold = _worst_fold_score(p8_07, p8_90, p8_120, train_df, splits)
    logger.info("  Plan8 weighted=%.6f  7d_ll=%.6f  worst_fold=%.6f", p8_weighted, p8_7d_ll, p8_worst_fold)

    d_weighted = p8_weighted - e6_weighted
    d_7d_ll = p8_7d_ll - e6_7d_ll
    pass_weighted = d_weighted >= E8_WEIGHTED_DELTA_MIN
    pass_7d_ll = d_7d_ll <= E8_7D_LL_DELTA_MAX
    pass_worst_fold = p8_worst_fold >= e6_worst_fold
    primary_pass = pass_weighted and pass_7d_ll and pass_worst_fold

    logger.info("\n" + "=" * 70)
    logger.info("E8 RESULT")
    logger.info("  ΔWeighted   = %.6f  (need >= +%.3f)  %s", d_weighted, E8_WEIGHTED_DELTA_MIN, "OK" if pass_weighted else "FAIL")
    logger.info("  Δ7dLL       = %.6f  (need <= %.3f)  %s", d_7d_ll, E8_7D_LL_DELTA_MAX, "OK" if pass_7d_ll else "FAIL")
    logger.info("  Worst-fold  = %.6f  (need >= %.6f)  %s", p8_worst_fold, e6_worst_fold, "OK" if pass_worst_fold else "FAIL")
    if primary_pass:
        logger.info("  PASS: Adopt Plan 8 (cohort mixture + multi-calibration).")
    else:
        logger.warning("  FAIL: Likely ceiling with current data; keep E6 winner.")
    logger.info("=" * 70)

    sys.exit(0 if primary_pass else 1)


if __name__ == "__main__":
    main()
