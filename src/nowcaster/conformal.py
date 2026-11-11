"""Conformal prediction intervals from a model's own past out-of-sample errors.

The interval for month t is the forecast plus or minus a quantile of the absolute errors
of months before t. No normal-distribution assumption. Coverage holds if errors are
exchangeable over time, which is checked on the backtest rather than assumed.
"""

import math

import numpy as np
import pandas as pd

LEVELS = (0.80, 0.95)
MIN_HISTORY = 36


def half_width(abs_errors: np.ndarray, level: float) -> float:
    """Split-conformal quantile: the ceil((n + 1) * level)-th smallest error.

    Infinite when there are too few errors for the level, as the method requires.
    """
    n = len(abs_errors)
    rank = math.ceil((n + 1) * level)
    if rank > n:
        return math.inf
    return float(np.sort(abs_errors)[rank - 1])


def online_half_widths(
    abs_errors: pd.Series, level: float, min_history: int = MIN_HISTORY
) -> pd.Series:
    """For each month after the first `min_history`, the half-width built from the errors
    of earlier months only."""
    values = abs_errors.to_numpy(dtype=float)
    widths = {
        period: half_width(values[:i], level)
        for i, period in enumerate(abs_errors.index)
        if i >= min_history
    }
    return pd.Series(widths, dtype=float)


def coverage(abs_errors: pd.Series, level: float) -> tuple[float, float, int]:
    """(share of months inside the interval, mean interval width, months evaluated)."""
    widths = online_half_widths(abs_errors, level)
    inside = abs_errors.loc[widths.index] <= widths
    return float(inside.mean()), float(2 * widths.mean()), len(widths)
