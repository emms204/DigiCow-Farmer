#!/usr/bin/env python3
"""
E10.3 — Calibration search (LL heads only).

For each target's LL head, try none / platt / isotonic / beta. Keep per-target
calibrator only if robust across folds (fold_std, worst_fold) and AUC drop
within tolerance. Uses same E10 harness and e10_baseline.json.

Usage:
  python e10_3_calibration_search.py
  python e10_3_calibration_search.py --run-baseline-if-missing
  python e10_3_calibration_search.py --auc-tol 0.01 --fold-std-factor 1.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.calibration import CalibrationMethod
from shared.constants import DATE_COL, TARGET_COLS
from shared.data_loader import DataLoader

from e10_dual_hpo import (
    E10_BASELINE_PATH,
    E10_N_SPLITS,
    E10_SEED,
    _load_baseline,
    run_single_e10_harness,
)
from e10_dual_hpo import _run_and_save_baseline_once  # noqa: F401  # used by getattr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

E10_3_RESULT_PATH = Path(__file__).resolve().parent / "e10_3_calibration.json"

# Calibrators to search (LL heads only)
CALIBRATORS = [
    CalibrationMethod.NONE,
    CalibrationMethod.PLATT,
    CalibrationMethod.ISOTONIC,
    CalibrationMethod.BETA,
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E10.3: Calibration search for LL heads (none/platt/isotonic/beta)"
    )
    parser.add_argument(
        "--run-baseline-if-missing",
        action="store_true",
        help="If e10_baseline.json missing, run one E10 harness and save it",
    )
    parser.add_argument(
        "--auc-tol",
        type=float,
        default=0.01,
        help="Max allowed drop in OOF AUC vs baseline (per target)",
    )
    parser.add_argument(
        "--fold-std-factor",
        type=float,
        default=1.2,
        help="Allow fold_std up to baseline_fold_std * this",
    )
    parser.add_argument(
        "--worst-fold-tol",
        type=float,
        default=0.01,
        help="Allow worst_fold down to baseline_worst_fold - this",
    )
    parser.add_argument("--n-splits", type=int, default=E10_N_SPLITS)
    args = parser.parse_args()

    os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = train_df[DATE_COL].astype("datetime64[ns]")
    prior_df = prior_df.copy()
    prior_df[DATE_COL] = prior_df[DATE_COL].astype("datetime64[ns]")

    if not E10_BASELINE_PATH.exists() and args.run_baseline_if_missing:
        _run_and_save_baseline_once(train_df, prior_df, n_splits=args.n_splits)

    baseline = _load_baseline()
    baseline_weighted = baseline["weighted_score"]
    baseline_07_auc = baseline["oof_07_auc"]
    baseline_90_auc = baseline["oof_90_auc"]
    baseline_120_auc = baseline["oof_120_auc"]
    baseline_fold_std = baseline["fold_std"]
    baseline_worst_fold = baseline["worst_fold_score"]

    logger.info("=" * 70)
    logger.info("E10.3 CALIBRATION SEARCH (LL heads only)")
    logger.info("  Baseline: weighted=%.6f, fold_std=%.6f, worst_fold=%.6f",
                baseline_weighted, baseline_fold_std, baseline_worst_fold)
    logger.info("  AUC tol=%.4f, fold_std factor=%.2f, worst_fold tol=%.4f",
                args.auc_tol, args.fold_std_factor, args.worst_fold_tol)
    logger.info("=" * 70)

    # Quiet per-fold logs during search
    _noisy = [logging.getLogger("shared.feature_engineering"), logging.getLogger("e10_dual_hpo")]
    _saved = [log.level for log in _noisy]
    for log in _noisy:
        log.setLevel(logging.WARNING)

    best_per_target: dict[str, str] = {}

    for target in TARGET_COLS:
        best_method = CalibrationMethod.ISOTONIC
        best_weighted = baseline_weighted

        for method in CALIBRATORS:
            ll_cal = {t: (method if t == target else CalibrationMethod.ISOTONIC) for t in TARGET_COLS}
            result = run_single_e10_harness(
                train_df,
                prior_df,
                n_splits=args.n_splits,
                seed=E10_SEED,
                ll_calibration_per_target=ll_cal,
            )

            # AUC drop guardrail
            if result["oof_07_auc"] < baseline_07_auc - args.auc_tol:
                continue
            if result["oof_90_auc"] < baseline_90_auc - args.auc_tol:
                continue
            if result["oof_120_auc"] < baseline_120_auc - args.auc_tol:
                continue

            # Fold robustness
            if result["fold_std"] > baseline_fold_std * args.fold_std_factor:
                continue
            if result["worst_fold_score"] < baseline_worst_fold - args.worst_fold_tol:
                continue

            if result["weighted_score"] > best_weighted:
                best_weighted = result["weighted_score"]
                best_method = method

        best_per_target[target] = best_method.name.lower()
        logger.info("  %s: best=%s weighted=%.6f", target, best_method.name, best_weighted)

    for log, level in zip(_noisy, _saved):
        log.setLevel(level)

    # Overall best weighted when using chosen calibrators
    ll_cal_final = {t: CalibrationMethod[best_per_target[t].upper()] for t in TARGET_COLS}
    result_final = run_single_e10_harness(
        train_df,
        prior_df,
        n_splits=args.n_splits,
        seed=E10_SEED,
        ll_calibration_per_target=ll_cal_final,
    )

    delta = result_final["weighted_score"] - baseline_weighted
    logger.info("")
    logger.info("=" * 70)
    logger.info("E10.3 RESULT")
    logger.info("  Calibrators: 7d=%s, 90d=%s, 120d=%s",
                best_per_target[TARGET_COLS[0]],
                best_per_target[TARGET_COLS[1]],
                best_per_target[TARGET_COLS[2]])
    logger.info("  Baseline weighted: %.6f", baseline_weighted)
    logger.info("  Final weighted:    %.6f", result_final["weighted_score"])
    logger.info("  Δ:                 %.6f", delta)
    logger.info("=" * 70)

    out = {
        "calibration_per_target": best_per_target,
        "baseline_weighted": baseline_weighted,
        "final_weighted": result_final["weighted_score"],
        "delta": delta,
    }
    E10_3_RESULT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("  Saved to %s", E10_3_RESULT_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
