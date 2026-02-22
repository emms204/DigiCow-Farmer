#!/usr/bin/env python3
"""
Master runner — execute all plans and compare run results.

Usage:
    python run_all.py              # run all plans
    python run_all.py 1 3          # run only Plan 1 and Plan 3
    python run_all.py --minimal-features   # run all 6 plans with minimal (9) features only, then evaluate
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan1.model import LGBMModel
from plan2.model import CatBoostModel
from plan3.model import StackingEnsemble
from plan4.model import DualOptimiser
from plan5.model import SimpleModel

try:
    from plan6.model import HazardNeuralModel

    HAS_PLAN6 = True
except Exception:
    HAS_PLAN6 = False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_all")

    parser = argparse.ArgumentParser(description="Run plans and optionally evaluate submissions")
    parser.add_argument(
        "plans",
        type=int,
        nargs="*",
        default=None,
        help="Plan numbers to run (default: all 1–6)",
    )
    parser.add_argument(
        "--minimal-features",
        action="store_true",
        help="Use minimal (9) features only for all plans; then run evaluate_submissions.py",
    )
    args = parser.parse_args()

    requested = (
        set(args.plans)
        if args.plans is not None and len(args.plans) > 0
        else {1, 2, 3, 4, 5, 6}
    )

    if args.minimal_features:
        os.environ["DIGICOW_MINIMAL_FEATURES"] = "1"
        logger.info("DIGICOW_MINIMAL_FEATURES=1 — all plans will use minimal (9) features only")

    plans: list[tuple[int, str, object]] = [
        (1, "Plan 1: LightGBM + Calibration", LGBMModel()),
        (2, "Plan 2: CatBoost + Combined Data", CatBoostModel()),
        (3, "Plan 3: Stacking Ensemble", StackingEnsemble()),
        (4, "Plan 4: Dual Optimisation", DualOptimiser()),
        (5, "Plan 5: Simple Calibrated", SimpleModel()),
    ]
    if HAS_PLAN6:
        plans.append((6, "Plan 6: Hazard Neural Network", HazardNeuralModel()))
    elif 6 in requested:
        logger.warning(
            "Plan 6 requested but unavailable (missing dependencies). Skipping."
        )

    results: list[tuple[str, str, float]] = []

    for plan_num, plan_name, model in plans:
        if plan_num not in requested:
            continue

        logger.info("\n" + "▶" * 30)
        logger.info(" Starting %s", plan_name)
        logger.info("▶" * 30 + "\n")

        t0 = time.perf_counter()
        try:
            path = model.run()
            elapsed = time.perf_counter() - t0
            results.append((plan_name, str(path), elapsed))
            logger.info("✓ %s finished in %.1f s → %s", plan_name, elapsed, path)
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception("✗ %s FAILED after %.1f s", plan_name, elapsed)
            results.append((plan_name, "FAILED", elapsed))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info(" SUMMARY")
    logger.info("=" * 70)
    for name, path, elapsed in results:
        status = "✓" if path != "FAILED" else "✗"
        logger.info("  %s %-35s  %6.1f s  %s", status, name, elapsed, path)
    logger.info("=" * 70)

    if args.minimal_features and results:
        logger.info("\nRunning evaluate_submissions.py on submissions ...")
        root = Path(__file__).resolve().parent
        code = subprocess.run(
            [sys.executable, str(root / "evaluate_submissions.py")],
            cwd=str(root),
        ).returncode
        if code != 0:
            logger.warning("evaluate_submissions.py exited with code %s", code)
        else:
            logger.info("evaluate_submissions.py finished.")


if __name__ == "__main__":
    main()
