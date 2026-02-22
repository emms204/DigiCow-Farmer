#!/usr/bin/env python3
"""
Plan 6 entry point — Hazard-style neural network.

Usage:
    python -m plan6.run
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan6.model import HazardNeuralModel


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    model = HazardNeuralModel()
    path = model.run(submission_filename="plan6_hazard_submission.csv")
    print(f"\n✅ Plan 6 submission saved to: {path}")


if __name__ == "__main__":
    main()

