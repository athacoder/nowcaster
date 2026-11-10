"""Baseline forecasters. Each takes the history of month-on-month changes (percent, indexed
by month, ending the month before the target) and returns a forecast for the next month.
"""

import warnings
from collections.abc import Callable

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

Forecaster = Callable[[pd.Series], float]


def seasonal_naive(history: pd.Series) -> float:
    """Same month last year. Equivalent to "year-on-year inflation stays where it was"."""
    return float(history.iloc[-12])


def seasonal_mean(history: pd.Series) -> float:
    """Average change of the same calendar month over every earlier year."""
    target_month = (history.index[-1] + 1).month
    return float(history[pd.PeriodIndex(history.index).month == target_month].mean())


def arima(history: pd.Series) -> float:
    """Airline model: SARIMA(0,1,1)(0,1,1)12 on the log index, i.e. MA terms on the
    seasonally differenced month-on-month change. The order is fixed, never tuned on the
    backtest."""
    values = history.to_numpy(dtype=float)
    model = SARIMAX(values, order=(0, 0, 1), seasonal_order=(0, 1, 1, 12), trend="n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp=False)
    return float(fitted.forecast(1)[0])


BASELINES: dict[str, Forecaster] = {
    "seasonal_naive": seasonal_naive,
    "seasonal_mean": seasonal_mean,
    "arima": arima,
}
