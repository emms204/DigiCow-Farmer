"""
Submission file generation for the DigiCow challenge.

Produces a CSV matching the expected format with columns:
    ID, Target_07_AUC, Target_90_AUC, Target_120_AUC,
    Target_07_LogLoss, Target_90_LogLoss, Target_120_LogLoss
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from shared.constants import (
    ID_COL,
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
    SUBMISSION_DIR,
    SUBMISSION_MAP,
    TARGET_COLS,
)

logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """Create a competition-ready submission CSV.

    Supports either *identical* probabilities for the AUC and LogLoss
    columns (standard) or *separate* optimised values (Plan 4 strategy).

    Parameters
    ----------
    sample_submission : pd.DataFrame
        The ``SampleSubmission.csv`` DataFrame to use as the template.
    output_dir : Path | str, optional
        Directory to write submissions.  Defaults to ``submissions/``.

    Example
    -------
    >>> gen = SubmissionGenerator(sample_df)
    >>> gen.generate(
    ...     test_ids=test_df["ID"],
    ...     predictions={"adopted_within_07_days": p07, ...},
    ...     filename="plan1_submission.csv",
    ... )
    """

    def __init__(
        self,
        sample_submission: pd.DataFrame,
        output_dir: Path | str | None = None,
    ) -> None:
        self.template = sample_submission.copy()
        self.output_dir = Path(output_dir) if output_dir else SUBMISSION_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        test_ids: pd.Series,
        predictions: dict[str, np.ndarray],
        filename: str = "submission.csv",
        predictions_auc: dict[str, np.ndarray] | None = None,
    ) -> Path:
        """Write a submission file.

        Parameters
        ----------
        test_ids : pd.Series
            The ID column from the test set, in order.
        predictions : dict[str, np.ndarray]
            Mapping from target name to predicted probabilities.
            Used for LogLoss columns (and AUC columns if *predictions_auc*
            is not supplied).
        filename : str
            Name for the output CSV.
        predictions_auc : dict[str, np.ndarray], optional
            If provided, these probabilities are written to the AUC columns
            while *predictions* goes to the LogLoss columns.  Enables the
            dual-optimisation strategy of Plan 4.

        Returns
        -------
        Path
            Absolute path to the written file.
        """
        sub = self.template.copy()
        sub[ID_COL] = test_ids.values

        for target_col in TARGET_COLS:
            auc_col, ll_col = SUBMISSION_MAP[target_col]
            probs_ll = np.clip(predictions[target_col], PROB_CLIP_MIN, PROB_CLIP_MAX)

            if predictions_auc is not None:
                probs_auc = np.clip(
                    predictions_auc[target_col], PROB_CLIP_MIN, PROB_CLIP_MAX
                )
            else:
                probs_auc = probs_ll

            sub[auc_col] = probs_auc
            sub[ll_col] = probs_ll

        out_path = self.output_dir / filename
        sub.to_csv(out_path, index=False)
        logger.info("Submission written to %s  (%d rows)", out_path, len(sub))
        return out_path

