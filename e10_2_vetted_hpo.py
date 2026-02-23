#!/usr/bin/env python3
"""
E10.2 — LGBM HPO with vetted feature set (single model per target).

Same as E10.1 but:
  - Vetted features: confidence_aggregates + recency_intensity (VETTED_FEATURE_GROUPS).
  - Guardrails: same as E10.1 + late-fold score not worse.
  - Pass rule: keep only if E10.2 best weighted > E10.1 best (from e10_1_best.json).

Prerequisite: Run E10.1 first so e10_1_best.json exists (or pass --e10-1-best SCORE).

Usage:
  python e10_2_vetted_hpo.py [--run-baseline-if-missing] [--study 7d|all] [--n-trials 30]
  python e10_2_vetted_hpo.py --e10-1-best 0.893  # if e10_1_best.json missing
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

# Import harness and defaults from E10
from e10_dual_hpo import (
    E10_N_SPLITS,
    E10_SEED,
    run_single_e10_harness,
    _default_ll_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

E10_2_BASELINE_PATH = Path(__file__).resolve().parent / "e10_2_baseline.json"
E10_1_BEST_PATH = Path(__file__).resolve().parent / "e10_1_best.json"


def _load_baseline() -> dict[str, Any]:
    if not E10_2_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"E10.2 baseline not found at {E10_2_BASELINE_PATH}. "
            "Run with --run-baseline-if-missing first."
        )
    return json.loads(E10_2_BASELINE_PATH.read_text(encoding="utf-8"))


def _run_and_save_baseline_once(
    train_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    n_splits: int = E10_N_SPLITS,
) -> dict[str, Any]:
    """One E10 harness run with vetted features; save as E10.2 baseline."""
    logger.info("Running one E10.2 harness (vetted features) to create baseline at %s ...", E10_2_BASELINE_PATH)
    result = run_single_e10_harness(
        train_df,
        prior_df,
        n_splits=n_splits,
        seed=E10_SEED,
        feature_groups=VETTED_FEATURE_GROUPS,
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
    E10_2_BASELINE_PATH.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info("  Baseline saved: weighted=%.6f last_fold=%.6f", result["weighted_score"], result["last_fold_score"])
    return payload


def _load_e10_1_best(args: argparse.Namespace) -> float:
    """E10.1 best weighted score for pass rule (E10.2 must beat this)."""
    if getattr(args, "e10_1_best", None) is not None:
        return float(args.e10_1_best)
    if not E10_1_BEST_PATH.exists():
        raise FileNotFoundError(
            f"E10.1 best not found at {E10_1_BEST_PATH}. Run E10.1 HPO first or pass --e10-1-best SCORE."
        )
    data = json.loads(E10_1_BEST_PATH.read_text(encoding="utf-8"))
    return float(data["weighted_score"])


def run_hpo(args: argparse.Namespace) -> int:
    """E10.2: LGBM dual HPO with vetted features, late-fold guardrail, pass vs E10.1."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Do NOT set DIGICOW_MINIMAL_FEATURES — we use VETTED_FEATURE_GROUPS
    if os.environ.get("DIGICOW_MINIMAL_FEATURES") == "1":
        os.environ.pop("DIGICOW_MINIMAL_FEATURES", None)

    loader = DataLoader()
    train_df, _test_df, prior_df, _ = loader.load_all()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    prior_df[DATE_COL] = pd.to_datetime(prior_df[DATE_COL])

    if not E10_2_BASELINE_PATH.exists() and getattr(args, "run_baseline_if_missing", False):
        _run_and_save_baseline_once(train_df, prior_df, n_splits=args.n_splits)
    baseline = _load_baseline()
    baseline_weighted = baseline["weighted_score"]
    baseline_7d_ll = baseline["oof_07_ll"]
    baseline_worst_fold = baseline["worst_fold_score"]
    baseline_fold_std = baseline["fold_std"]
    baseline_last_fold = baseline["last_fold_score"]

    e10_1_best_score = _load_e10_1_best(args)

    logger.info("=" * 70)
    logger.info("E10.2 LGBM DUAL HPO (vetted features)")
    logger.info("  Baseline weighted: %.6f (from %s)", baseline_weighted, E10_2_BASELINE_PATH)
    logger.info("  E10.1 best (must beat): %.6f", e10_1_best_score)
    logger.info("  Guardrails: 7d_ll, worst_fold, fold_std, late-fold (last_fold) not worse")
    logger.info("  Pass rule: E10.2 best > E10.1 best (%.6f)", e10_1_best_score)
    logger.info("=" * 70)

    # Quiet per-fold and feature-engineering logs during HPO (summary only)
    _noisy_loggers = [
        logging.getLogger("shared.feature_engineering"),
        logging.getLogger("e10_dual_hpo"),
    ]
    _saved_levels = [log.level for log in _noisy_loggers]
    for log in _noisy_loggers:
        log.setLevel(logging.WARNING)

    def _make_objective(study_key: str):
        def objective(trial: optuna.Trial) -> float:
            ll_params_per_target = {t: _default_ll_params(E10_SEED) for t in TARGET_COLS}
            num_rounds = trial.suggest_int("num_boost_round", 300, 1500)
            tuned = {
                **_default_ll_params(E10_SEED),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.08),
                "num_leaves": trial.suggest_int("num_leaves", 15, 45),
                "max_depth": trial.suggest_int("max_depth", 4, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 2.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.5, 3.0),
            }
            if study_key == "7d":
                ll_params_per_target[TARGET_COLS[0]] = tuned
            elif study_key == "90d":
                ll_params_per_target[TARGET_COLS[1]] = tuned
            else:
                ll_params_per_target[TARGET_COLS[2]] = tuned

            result = run_single_e10_harness(
                train_df,
                prior_df,
                n_splits=args.n_splits,
                seed=E10_SEED,
                ll_params_per_target=ll_params_per_target,
                num_boost_round=num_rounds,
                feature_groups=VETTED_FEATURE_GROUPS,
            )
            score = result["weighted_score"]
            # Guardrails (same as E10.1 + late-fold)
            if result["oof_07_ll"] > baseline_7d_ll + 1e-6:
                return 0.0
            if result["worst_fold_score"] < baseline_worst_fold - 1e-6:
                return 0.0
            max_std = baseline_fold_std * 1.5 + 0.01
            if result["fold_std"] > max_std:
                return 0.0
            if result["last_fold_score"] < baseline_last_fold - 1e-6:
                return 0.0
            return score

        return objective

    studies_to_run = (
        ["7d", "90d", "120d"]
        if args.study == "all"
        else [args.study]
    )

    best_overall = baseline_weighted
    best_params: dict[str, dict] = {}

    for sk in studies_to_run:
        logger.info("\n--- E10.2 Optuna study: %s (n_trials=%d) ---", sk, args.n_trials)
        study = optuna.create_study(direction="maximize", study_name=f"e10_2_{sk}")
        study.optimize(_make_objective(sk), n_trials=args.n_trials, show_progress_bar=True)
        if study.best_trial is not None and study.best_value > baseline_weighted:
            logger.info(
                "  Best %s: weighted=%.6f (Δ=%.6f)",
                sk,
                study.best_value,
                study.best_value - baseline_weighted,
            )
            if study.best_value > best_overall:
                best_overall = study.best_value
            best_params[sk] = study.best_params
        else:
            logger.info("  No improvement over E10.2 baseline for %s", sk)

    # Restore log levels
    for log, level in zip(_noisy_loggers, _saved_levels):
        log.setLevel(level)

    # Pass rule: keep E10.2 only if better than E10.1
    beats_e10_1 = best_overall > e10_1_best_score
    delta_vs_e10_1 = best_overall - e10_1_best_score

    logger.info("\n" + "=" * 70)
    logger.info("E10.2 RESULT")
    logger.info("  E10.2 baseline weighted: %.6f", baseline_weighted)
    logger.info("  E10.2 best weighted:     %.6f", best_overall)
    logger.info("  E10.1 best weighted:     %.6f", e10_1_best_score)
    logger.info("  Δ vs E10.1:             %.6f", delta_vs_e10_1)
    logger.info("  Pass (E10.2 > E10.1):   %s", "PASS" if beats_e10_1 else "FAIL")
    if best_params:
        logger.info("  Best params keys: %s", list(best_params.keys()))
    logger.info("=" * 70)

    return 0 if beats_e10_1 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E10.2: LGBM dual HPO with vetted features (keep only if better than E10.1)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials per study",
    )
    parser.add_argument(
        "--study",
        type=str,
        default="7d",
        choices=["7d", "90d", "120d", "all"],
        help="Which head(s) to tune",
    )
    parser.add_argument("--n-splits", type=int, default=E10_N_SPLITS)
    parser.add_argument(
        "--run-baseline-if-missing",
        action="store_true",
        help="If e10_2_baseline.json is missing, run one vetted-features harness and save it",
    )
    parser.add_argument(
        "--e10-1-best",
        type=float,
        default=None,
        metavar="SCORE",
        help="E10.1 best weighted score (if e10_1_best.json not found)",
    )
    args = parser.parse_args()

    sys.exit(run_hpo(args))


if __name__ == "__main__":
    main()
