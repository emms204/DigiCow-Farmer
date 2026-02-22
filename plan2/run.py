#!/usr/bin/env python3
"""
Plan 2 entry point — CatBoost + Combined Prior/Train data.

Usage:
    python -m plan2.run
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan2.model import CatBoostModel


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    model = CatBoostModel()
    path = model.run(submission_filename="plan2_catboost_submission.csv")
    print(f"\n✅ Plan 2 submission saved to: {path}")


if __name__ == "__main__":
    main()

