"""Decisions for the daily run: which month to predict and which to score.

Kept apart from the I/O in `cli._daily` so each rule has a test.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

Released = Callable[[pd.Period], bool]


def predicted_months(predictions_dir: Path) -> list[pd.Period]:
    return sorted(
        pd.Period(path.stem, freq="M")
        for path in predictions_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].json")
    )


def target_month(predictions_dir: Path, released: Released) -> pd.Period:
    """The first month after the last prediction whose CPI is not out yet.

    A month whose CPI came out before it was predicted is skipped, not predicted late.
    """
    month = predicted_months(predictions_dir)[-1] + 1
    while released(month):
        month += 1
    return month


def unscored(predictions_dir: Path) -> list[pd.Period]:
    """Predicted months with no actual value in `record.csv` yet."""
    record_path = predictions_dir / "record.csv"
    scored: set[str] = set()
    if record_path.exists():
        record = pd.read_csv(record_path, dtype={"month": str})
        scored = set(record.loc[record["actual"].notna(), "month"])
    return [month for month in predicted_months(predictions_dir) if str(month) not in scored]
