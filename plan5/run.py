#!/usr/bin/env python3
"""
Plan 5 entry point — Simple Calibrated Model.

Usage:
    python -m plan5.run
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan5.model import SimpleModel


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    model = SimpleModel()
    path = model.run(submission_filename="plan5_simple_submission.csv")
    print(f"\n✅ Plan 5 submission saved to: {path}")


if __name__ == "__main__":
    main()

