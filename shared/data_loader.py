"""
Data loading and normalisation for DigiCow challenge datasets.

Handles the format differences between Train/Test (nested Python lists for
trainer and topics) and Prior (flat strings), and exposes a unified API.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.constants import (
    DATA_DIR,
    DATE_COL,
    FARMER_COL,
    ID_COL,
    PRIOR_FILE,
    SAMPLE_SUBMISSION_FILE,
    TARGET_COLS,
    TEST_FILE,
    TOPICS_COL,
    TRAIN_FILE,
    TRAINER_COL,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Load, validate, and normalise the competition CSV files.

    All heavy parsing (date conversion, list-literal evaluation) is performed
    once at load time so downstream consumers receive clean DataFrames.

    Parameters
    ----------
    data_dir : Path | str, optional
        Root directory containing the CSV files.  Defaults to the project
        ``DATA_DIR`` constant.

    Example
    -------
    >>> loader = DataLoader()
    >>> train, test, prior, sample = loader.load_all()
    """

    def __init__(self, data_dir: Optional[Path | str] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    # ── public API ─────────────────────────────────────────────────────

    def load_all(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and normalise all four competition files.

        Returns
        -------
        train, test, prior, sample_submission : tuple[DataFrame, ...]
            Each DataFrame has dates parsed, trainer normalised to a plain
            string, and topics normalised to a flat ``list[str]``.
        """
        train = self._load_and_normalise(TRAIN_FILE, has_targets=True)
        test = self._load_and_normalise(TEST_FILE, has_targets=False)
        prior = self._load_and_normalise(PRIOR_FILE, has_targets=True)
        sample = pd.read_csv(self.data_dir / SAMPLE_SUBMISSION_FILE)

        self._log_shapes(train, test, prior)
        return train, test, prior, sample

    def load_train(self) -> pd.DataFrame:
        """Load only the training set."""
        return self._load_and_normalise(TRAIN_FILE, has_targets=True)

    def load_test(self) -> pd.DataFrame:
        """Load only the test set."""
        return self._load_and_normalise(TEST_FILE, has_targets=False)

    def load_prior(self) -> pd.DataFrame:
        """Load only the prior (historical) dataset."""
        return self._load_and_normalise(PRIOR_FILE, has_targets=True)

    # ── internal helpers ───────────────────────────────────────────────

    def _load_and_normalise(
        self, filename: str, *, has_targets: bool
    ) -> pd.DataFrame:
        """Read a CSV and apply all normalisation steps.

        Complexity: O(n) for n rows – each row is visited once per column
        that needs parsing.
        """
        path = self.data_dir / filename
        logger.info("Loading %s …", path)
        df = pd.read_csv(path)

        # Parse date column
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

        # Normalise trainer to a single string (most rows have one trainer)
        df[TRAINER_COL] = df[TRAINER_COL].apply(self._normalise_trainer)

        # Normalise topics to a flat list of strings
        df[TOPICS_COL] = df[TOPICS_COL].apply(self._normalise_topics)

        # Cast target columns to int where present
        if has_targets:
            for col in TARGET_COLS:
                if col in df.columns:
                    df[col] = df[col].astype(int)

        logger.info("  → %d rows, %d columns", *df.shape)
        return df

    @staticmethod
    def _normalise_trainer(raw: str) -> str:
        """Extract the primary trainer ID from varying formats.

        Train/Test format : ``"['TRA_szrwyfzz']"``
        Prior format      : ``"TRA_szrwyfzz"``

        Returns the first trainer ID as a plain string.
        Handles multi-trainer rows like ``"['TRA_a', 'TRA_b']"`` by
        returning a comma-joined string to preserve information.
        """
        if pd.isna(raw):
            return "UNKNOWN"
        raw = str(raw).strip()
        # Already a plain ID
        if raw.startswith("TRA_"):
            return raw
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return ",".join(str(t).strip() for t in parsed)
            return str(parsed)
        except (ValueError, SyntaxError):
            return raw

    @staticmethod
    def _normalise_topics(raw: str) -> list[str]:
        """Flatten nested topic lists into a deduplicated sorted list.

        Train/Test format : ``"[['Topic A', 'Topic B'], ['Topic C']]"``
        Prior format      : ``"['Topic A', 'Topic B']"``

        Returns a flat ``list[str]`` of unique topic names.
        """
        if pd.isna(raw):
            return []
        try:
            parsed = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return []

        topics: set[str] = set()
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, list):
                    topics.update(str(t).strip() for t in item)
                else:
                    topics.add(str(item).strip())
        return sorted(topics)

    @staticmethod
    def _log_shapes(
        train: pd.DataFrame, test: pd.DataFrame, prior: pd.DataFrame
    ) -> None:
        """Log summary of loaded datasets."""
        logger.info(
            "Loaded — Train: %s, Test: %s, Prior: %s",
            train.shape,
            test.shape,
            prior.shape,
        )

