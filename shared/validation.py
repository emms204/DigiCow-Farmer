"""
Cross-validation strategies for the DigiCow challenge.

Provides time-based and stratified splitting with consistent evaluation
of both Log Loss (75 %) and ROC-AUC (25 %) as per the competition metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from shared.constants import (
    CV_STRATEGY_DEFAULT,
    CV_TIME_CUTOFF_DEFAULT,
    DATE_COL,
    RANDOM_SEED,
    TARGET_COLS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldResult:
    """Immutable container for a single fold's evaluation scores."""

    fold: int
    target: str
    log_loss: float
    roc_auc: float

    @property
    def combined_score(self) -> float:
        """Weighted competition metric: 0.75 * LogLoss + 0.25 * (1 - AUC).

        Lower is better for LogLoss; higher is better for AUC.  We follow
        the competition definition where the *final score* blends both.
        For internal tracking we return the weighted loss form so that
        *lower is better* consistently.
        """
        return 0.75 * self.log_loss + 0.25 * (1.0 - self.roc_auc)


@dataclass
class CVResult:
    """Aggregated cross-validation results across folds and targets."""

    fold_results: list[FoldResult] = field(default_factory=list)

    def add(self, result: FoldResult) -> None:
        self.fold_results.append(result)

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising mean ± std per target."""
        rows = []
        all_targets = sorted({r.target for r in self.fold_results})
        targets_to_show = all_targets if all_targets else TARGET_COLS
        for target in targets_to_show:
            target_folds = [r for r in self.fold_results if r.target == target]
            if not target_folds:
                continue
            ll = [r.log_loss for r in target_folds]
            auc = [r.roc_auc for r in target_folds]
            combined = [r.combined_score for r in target_folds]
            rows.append(
                {
                    "target": target,
                    "mean_log_loss": np.mean(ll),
                    "std_log_loss": np.std(ll),
                    "mean_roc_auc": np.mean(auc),
                    "std_roc_auc": np.std(auc),
                    "mean_combined": np.mean(combined),
                }
            )
        return pd.DataFrame(rows)


class Validator:
    """Cross-validation orchestrator.

    Supports both stratified K-fold and a time-based holdout split.

    Parameters
    ----------
    n_splits : int
        Number of folds for stratified CV.
    seed : int
        Random seed.
    """

    def __init__(
        self, n_splits: int = 5, seed: int = RANDOM_SEED
    ) -> None:
        self.n_splits = n_splits
        self.seed = seed

    # ── Stratified K-Fold ──────────────────────────────────────────────

    def stratified_splits(
        self, y: pd.Series
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_idx, val_idx) arrays for stratified K-fold.

        O(n) per fold for indexing.
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
            yield train_idx, val_idx

    def cv_splits(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        strategy: str = CV_STRATEGY_DEFAULT,
        cutoff: str = CV_TIME_CUTOFF_DEFAULT,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield cross-validation splits using the requested strategy."""
        strategy_name = strategy.strip().lower()

        if strategy_name == "stratified":
            yield from self.stratified_splits(y)
            return

        if strategy_name == "time":
            train_idx, val_idx = self.time_based_split(df, cutoff=cutoff)
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError(
                    f"Time split produced empty partition with cutoff={cutoff}"
                )
            yield train_idx, val_idx
            return

        raise ValueError(
            f"Unknown CV strategy '{strategy}'. Use 'stratified' or 'time'."
        )

    # ── Time-based holdout ─────────────────────────────────────────────

    @staticmethod
    def time_based_split(
        df: pd.DataFrame, cutoff: str = "2025-01-01"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split by date: everything before *cutoff* is train, rest is val.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the ``training_day`` column (datetime).
        cutoff : str
            ISO date string for the split point.

        Returns
        -------
        train_idx, val_idx : tuple[np.ndarray, np.ndarray]
        """
        cutoff_dt = pd.Timestamp(cutoff)
        mask = df[DATE_COL] < cutoff_dt
        return (
            np.where(mask)[0],
            np.where(~mask)[0],
        )

    # ── Evaluation ─────────────────────────────────────────────────────

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_name: str = "",
        fold: int = 0,
        clip: tuple[float, float] = (0.001, 0.999),
    ) -> FoldResult:
        """Compute Log Loss and ROC-AUC for a single target/fold.

        Parameters
        ----------
        y_true : array-like of {0, 1}
        y_prob : array-like of floats in [0, 1]
        target_name : str
        fold : int
        clip : tuple
            Min/max bounds for predicted probabilities.

        Returns
        -------
        FoldResult
        """
        y_prob = np.clip(y_prob, *clip)

        ll = log_loss(y_true, y_prob)

        # ROC-AUC requires both classes present
        if len(np.unique(y_true)) < 2:
            auc = 0.5
            logger.warning(
                "Only one class in fold %d for %s — AUC set to 0.5",
                fold,
                target_name,
            )
        else:
            auc = roc_auc_score(y_true, y_prob)

        result = FoldResult(fold=fold, target=target_name, log_loss=ll, roc_auc=auc)
        logger.info(
            "  Fold %d | %s | LogLoss=%.5f  AUC=%.5f  Combined=%.5f",
            fold,
            target_name,
            ll,
            auc,
            result.combined_score,
        )
        return result
