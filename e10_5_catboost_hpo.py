#!/usr/bin/env python3
"""
E10.5 — CatBoost HPO with vetted features (single model per target).

Same protocol as E10.2: one model per target, weighted + 7d LL. Guardrails:
7d_ll, worst_fold, fold_std, last_fold not worse. Pass rule: E10.5 best > E10.1 best.

HPO: depth, learning_rate, l2_leaf_reg, bagging_temperature, random_strength,
border_count, iterations (Logloss objective).

Usage:
  python e10_5_catboost_hpo.py [--run-baseline-if-missing] [--study 7d|all] [--n-trials 30]
  python e10_5_catboost_hpo.py --e10-1-best 0.893
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.constants import DATE_COL, TARGET_COLS
from shared.data_loader import DataLoader
from shared.feature_engineering import VETTED_FEATURE_GROUPS

from e10_dual_hpo import (
    E10_N_SPLITS,
    E10_SEED,
    run_single_e10_harness_catboost,
    _default_catboost_ll_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

E10_5_BASELINE_PATH = Path(__file__).resolve().parent / "e10_5_baseline.json"
E10_5_BEST_PATH = Path(__file__).resolve().parent / "e10_5_best.json"
E10_1_BEST_PATH = Path(__file__).resolve().parent / "e10_1_best.json"


def _load_baseline() -> dict[str, Any]:
    if not E10_5_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"E10.5 baseline not found at {E10_5_BASELINE_PATH}. "
            "Run with --run-baseline-if-missing first."
        )
    return json.loads(E10_5_BASELINE_PATH.read_text(encoding="utf-8"))


def _run_and_save_baseline_once(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E10_N_SPLITS,
) -> dict[str, Any]:
    """One E10.5 CatBoost harness run; save as baseline."""
    logger.info("Running one E10.5 CatBoost harness to create baseline at %s ...", E10_5_BASELINE_PATH)
    result = run_single_e10_harness_catboost(
        train_df, prior_df, n_splits=n_splits, seed=E10_SEED,
    )
    payload = {
        "weighted_score": result["weighted_score"],
        "oof_07_ll": result["oof_07_ll"],
        "oof_07_auc": result["oof_07_auc"],
        "oof_90_ll": result["oof_90_ll"],
        "oof_90_auc": result["oof_90_auc"],
        "oof_120_ll": result["oof_120_ll"],
        "oof_120_auc": result["oof_120_auc"],
        "per_fold_weighted_scores": result["per_fold_weighted_scores"],
        "fold_std": result["fold_std"],
        "worst_fold_score": result["worst_fold_score"],
        "last_fold_score": result["last_fold_score"],
    }
    E10_5_BASELINE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("  Baseline saved: weighted=%.6f", result["weighted_score"])
    return payload


def _load_reference_score(args: argparse.Namespace) -> float:
    """E10.1 best for pass rule: E10.5 must beat this for blend."""
    if getattr(args, "e10_1_best", None) is not None:
        return float(args.e10_1_best)
    if E10_1_BEST_PATH.exists():
        return float(json.loads(E10_1_BEST_PATH.read_text(encoding="utf-8"))["weighted_score"])
    raise FileNotFoundError(
        f"Reference not found at {E10_1_BEST_PATH}. Run E10.1 first or pass --e10-1-best SCORE."
    )


def run_hpo(args: argparse.Namespace) -> int:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if os.environ.get("DIGICOW_MINIMAL_FEATURES") == "1":
        os.environ.pop("DIGICOW_MINIMAL_FEATURES", None)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    if not E10_5_BASELINE_PATH.exists() and getattr(args, "run_baseline_if_missing", False):
        _run_and_save_baseline_once(train_df, prior_df, n_splits=args.n_splits)
    baseline = _load_baseline()
    baseline_weighted = baseline["weighted_score"]
    baseline_7d_ll = baseline["oof_07_ll"]
    baseline_worst_fold = baseline["worst_fold_score"]
    baseline_fold_std = baseline["fold_std"]
    baseline_last_fold = baseline["last_fold_score"]
    reference_score = _load_reference_score(args)

    logger.info("=" * 70)
    logger.info("E10.5 CatBoost HPO (vetted features, single model per target)")
    logger.info("  Baseline weighted: %.6f (from %s)", baseline_weighted, E10_5_BASELINE_PATH)
    logger.info("  Reference (must beat for blend): %.6f", reference_score)
    logger.info("  Guardrails: 7d_ll, worst_fold, fold_std, last_fold not worse")
    logger.info("=" * 70)

    _noisy = [logging.getLogger("shared.feature_engineering"), logging.getLogger("e10_dual_hpo")]
    _saved = [log.level for log in _noisy]
    for log in _noisy:
        log.setLevel(logging.WARNING)

    def _suggest_catboost_params(trial: optuna.Trial, target_key: str) -> dict:
        p = _default_catboost_ll_params(E10_SEED).copy()
        p["depth"] = trial.suggest_int(f"{target_key}_depth", 4, 10)
        p["learning_rate"] = trial.suggest_float(f"{target_key}_lr", 0.02, 0.15)
        p["l2_leaf_reg"] = trial.suggest_float(f"{target_key}_l2_leaf_reg", 0.5, 10.0)
        p["bagging_temperature"] = trial.suggest_float(f"{target_key}_bagging_temperature", 0.0, 1.0)
        p["random_strength"] = trial.suggest_float(f"{target_key}_random_strength", 0.0, 1.0)
        p["border_count"] = trial.suggest_int(f"{target_key}_border_count", 32, 255)
        p["iterations"] = trial.suggest_int(f"{target_key}_iterations", 300, 1200)
        return p

    def make_objective(study_key: str):
        target_idx = {"7d": 0, "90d": 1, "120d": 2}[study_key]
        target = TARGET_COLS[target_idx]

        def objective(trial: optuna.Trial) -> float:
            ll_params = {t: _default_catboost_ll_params(E10_SEED) for t in TARGET_COLS}
            ll_params[target] = _suggest_catboost_params(trial, study_key)
            result = run_single_e10_harness_catboost(
                train_df,
                prior_df,
                n_splits=args.n_splits,
                seed=E10_SEED,
                ll_params_per_target=ll_params,
                feature_groups=VETTED_FEATURE_GROUPS,
            )
            if result["oof_07_ll"] > baseline_7d_ll + 1e-6:
                return 0.0
            if result["worst_fold_score"] < baseline_worst_fold - 1e-6:
                return 0.0
            if result["fold_std"] > baseline_fold_std * 1.5 + 0.01:
                return 0.0
            if result["last_fold_score"] < baseline_last_fold - 1e-6:
                return 0.0
            return result["weighted_score"]

        return objective

    studies = (
        ["7d", "90d", "120d"]
        if args.study == "all"
        else [args.study]
    )
    best_overall = baseline_weighted
    best_params: dict[str, Any] = {}

    for sk in studies:
        logger.info("\n--- E10.5 Optuna study: %s (n_trials=%d) ---", sk, args.n_trials)
        study = optuna.create_study(direction="maximize", study_name=f"e10_5_{sk}")
        study.optimize(make_objective(sk), n_trials=args.n_trials, show_progress_bar=True)
        if study.best_trial is not None and study.best_value > baseline_weighted:
            logger.info("  Best %s: weighted=%.6f", sk, study.best_value)
            if study.best_value > best_overall:
                best_overall = study.best_value
            best_params[sk] = study.best_params
        else:
            logger.info("  No improvement over E10.5 baseline for %s", sk)

    for log, level in zip(_noisy, _saved):
        log.setLevel(level)

    beats_ref = best_overall > reference_score
    logger.info("\n" + "=" * 70)
    logger.info("E10.5 RESULT")
    logger.info("  E10.5 baseline: %.6f  E10.5 best: %.6f  Reference: %.6f",
                baseline_weighted, best_overall, reference_score)
    logger.info("  Pass (orthogonal gain in blend): %s", "PASS" if beats_ref else "FAIL")
    logger.info("=" * 70)

    if best_params:
        out = {
            "weighted_score": best_overall,
            "best_params": best_params,
            "reference_score": reference_score,
        }
        E10_5_BEST_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        logger.info("  Saved best to %s", E10_5_BEST_PATH)

    return 0 if beats_ref else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E10.5: CatBoost dual HPO with vetted features (keep if orthogonal gain in blend)"
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--study",
        type=str,
        default="7d",
        choices=["7d", "90d", "120d", "all"],
    )
    parser.add_argument("--n-splits", type=int, default=E10_N_SPLITS)
    parser.add_argument("--run-baseline-if-missing", action="store_true")
    parser.add_argument("--e10-1-best", type=float, default=None, metavar="SCORE")
    args = parser.parse_args()
    sys.exit(run_hpo(args))


if __name__ == "__main__":
    main()
