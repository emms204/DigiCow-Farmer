#!/usr/bin/env python3
"""
Plan 1 entry point — LightGBM + Feature Engineering + Calibration.

Usage:
    python -m plan1.run
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan1.model import LGBMModel


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    model = LGBMModel()
    path = model.run(submission_filename="plan1_lgbm_submission.csv")
    print(f"\n✅ Plan 1 submission saved to: {path}")


if __name__ == "__main__":
    main()

