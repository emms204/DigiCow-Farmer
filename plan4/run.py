#!/usr/bin/env python3
"""
Plan 4 entry point — Dual-Optimisation Strategy.

Usage:
    python -m plan4.run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan4.model import DualOptimiser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-train", action="store_true", help="Write train (OOF) submission to submissions/train/")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    optimiser = DualOptimiser()
    path = optimiser.run(
        submission_filename="plan4_dual_submission.csv",
        write_train_submission=args.write_train,
    )
    print(f"\n✅ Plan 4 submission saved to: {path}")


if __name__ == "__main__":
    main()

