#!/usr/bin/env python3
"""
E10.1 with RandomSampler — Same as e10_dual_hpo.py HPO but uses random search.

Use this to compare TPE (default) vs random search: you should see more
variation in scores trial-to-trial and no "sticky" best value for many trials.

Uses the same e10_baseline.json and e10_1_best.json as e10_dual_hpo.py
(run baseline once with either script).

Usage:
  python e10_dual_hpo_random.py baseline [--runs 3] [--dry-run]   # same as E10.0
  python e10_dual_hpo_random.py hpo [--n-trials 50] [--study 7d_ll]  # RandomSampler
  python e10_dual_hpo_random.py hpo --run-baseline-if-missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import optuna

from e10_dual_hpo import E10_SEED, run_baseline, run_hpo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E10.1 LGBM dual HPO with Optuna RandomSampler (random search)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_baseline = sub.add_parser("baseline", help="E10.0: Lock benchmark (same as e10_dual_hpo)")
    p_baseline.add_argument("--runs", type=int, default=3)
    p_baseline.add_argument("--threshold", type=float, default=0.002)
    p_baseline.add_argument("--dry-run", action="store_true")
    p_baseline.add_argument("--n-splits", type=int, default=5)

    p_hpo = sub.add_parser("hpo", help="E10.1: LGBM dual HPO with RandomSampler")
    p_hpo.add_argument("--n-trials", type=int, default=30)
    p_hpo.add_argument(
        "--study",
        type=str,
        default="7d_ll",
        choices=["7d_ll", "7d_auc", "90d_ll", "90d_auc", "120d_ll", "120d_auc", "all"],
    )
    p_hpo.add_argument("--n-splits", type=int, default=5)
    p_hpo.add_argument("--run-baseline-if-missing", action="store_true")

    args = parser.parse_args()

    if args.command == "baseline":
        sys.exit(run_baseline(args))
    if args.command == "hpo":
        sampler = optuna.samplers.RandomSampler(seed=E10_SEED)
        sys.exit(run_hpo(args, sampler=sampler))
    sys.exit(1)


if __name__ == "__main__":
    main()
