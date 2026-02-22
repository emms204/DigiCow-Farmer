#!/usr/bin/env python3
"""
Plan 3 entry point — Stacking Ensemble.

Usage:
    python -m plan3.run
    python -m plan3.run --write-train   # also write submissions/train/ for evaluate_submissions.py --train
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan3.model import StackingEnsemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-train", action="store_true", help="Write train (OOF) submission to submissions/train/")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ensemble = StackingEnsemble()
    path = ensemble.run(
        submission_filename="plan3_stacking_submission.csv",
        write_train_submission=args.write_train,
    )
    print(f"\n✅ Plan 3 submission saved to: {path}")


if __name__ == "__main__":
    main()

