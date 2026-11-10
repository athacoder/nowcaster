"""Expanding-window backtest and the test used to compare two forecasters."""

import math

import numpy as np
import pandas as pd
from scipy import stats

from nowcaster.baselines import Forecaster


def run(series: pd.Series, models: dict[str, Forecaster], first: pd.Period) -> pd.DataFrame:
    """One-step-ahead forecasts for every month from `first` on.

    The forecast for month t sees `series` strictly before t and nothing else. That matches
    a forecast made at the end of month t: the CPI for t-1 came out around the 12th of t.
    Returns a frame indexed by month with `actual` and one column per model.
    """
    rows = []
    for target in series.index[series.index >= first]:
        history = series[series.index < target]
        row: dict[str, object] = {"period": target, "actual": float(series[target])}
        for name, model in models.items():
            row[name] = model(history)
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")


def scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE per model, in the units of the series (percentage points)."""
    errors = forecasts.drop(columns="actual").sub(forecasts["actual"], axis=0)
    return pd.DataFrame({"mae": errors.abs().mean(), "rmse": np.sqrt((errors**2).mean())})


def diebold_mariano(actual: pd.Series, a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Diebold-Mariano test on absolute error, one step ahead, with the Harvey-Leybourne-
    Newbold small-sample correction. Returns (statistic, two-sided p-value). A negative
    statistic means `a` has the smaller error."""
    d = ((a - actual).abs() - (b - actual).abs()).to_numpy(dtype=float)
    n = len(d)
    variance = d.var(ddof=0) / n  # h = 1: no autocovariance terms
    if variance == 0:
        return 0.0, 1.0
    statistic = d.mean() / math.sqrt(variance) * math.sqrt((n - 1) / n)
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p_value)
