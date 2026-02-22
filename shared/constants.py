"""
Global constants for the DigiCow Farmer Training Adoption Challenge.

Centralises all magic strings, column names, paths, and hyperparameter
defaults so they are defined once and imported everywhere.
"""

from pathlib import Path
from typing import Final

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR
SUBMISSION_DIR: Final[Path] = BASE_DIR / "submissions"

TRAIN_FILE: Final[str] = "Train.csv"
TEST_FILE: Final[str] = "Test.csv"
PRIOR_FILE: Final[str] = "Prior.csv"
SAMPLE_SUBMISSION_FILE: Final[str] = "SampleSubmission.csv"

# ── Target columns ─────────────────────────────────────────────────────────
TARGET_COLS: Final[list[str]] = [
    "adopted_within_07_days",
    "adopted_within_90_days",
    "adopted_within_120_days",
]

# ── Submission column mapping (target → submission columns) ────────────────
SUBMISSION_MAP: Final[dict[str, tuple[str, str]]] = {
    "adopted_within_07_days":  ("Target_07_AUC", "Target_07_LogLoss"),
    "adopted_within_90_days":  ("Target_90_AUC", "Target_90_LogLoss"),
    "adopted_within_120_days": ("Target_120_AUC", "Target_120_LogLoss"),
}

# ── Feature groups ─────────────────────────────────────────────────────────
CATEGORICAL_FEATURES: Final[list[str]] = [
    "gender",
    "registration",
    "age",
    "county",
    "subcounty",
    "ward",
]

BINARY_FEATURES: Final[list[str]] = [
    "belong_to_cooperative",
    "has_topic_trained_on",
]

ID_COL: Final[str] = "ID"
FARMER_COL: Final[str] = "farmer_name"
DATE_COL: Final[str] = "training_day"
TRAINER_COL: Final[str] = "trainer"
TOPICS_COL: Final[str] = "topics_list"
GROUP_COL: Final[str] = "group_name"

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED: Final[int] = 42

# ── Cross-validation defaults ──────────────────────────────────────────────
CV_STRATEGY_DEFAULT: Final[str] = "time"  # {"time", "stratified"}
CV_TIME_CUTOFF_DEFAULT: Final[str] = "2025-01-01"

# ── Probability clipping (prevents Log-Loss explosion) ─────────────────────
PROB_CLIP_MIN: Final[float] = 0.001
PROB_CLIP_MAX: Final[float] = 0.999

# ── Topic category keywords ────────────────────────────────────────────────
TOPIC_CATEGORIES: Final[dict[str, list[str]]] = {
    "dairy": [
        "dairy", "milk", "cow", "calf", "lactating", "calving",
        "herd", "breeding", "silage", "fodder",
    ],
    "poultry": [
        "poultry", "chicken", "kienyeji", "layers", "broiler",
        "biosecurity", "egg",
    ],
    "crop": [
        "maize", "bean", "seed", "weed", "fertilizer", "pest",
        "crop", "topdressing", "planting",
    ],
    "general": [
        "record", "hygiene", "ppe", "finance", "app", "ndume",
        "management", "vaccination", "health",
    ],
}
