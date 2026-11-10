"""Forecasters that use price features.

Each takes (history, past, now): the target's history up to the month before the target,
the feature rows for months in that history, and the feature row for the target month.
The list was fixed before any of them was scored.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nowcaster import baselines
from nowcaster.features import VEGETABLES

FeatureForecaster = Callable[[pd.Series, pd.DataFrame, pd.Series], float]
ALPHAS = np.logspace(-1, 3, 9)


def _ridge_correction(residual: pd.Series, past: pd.DataFrame, now: pd.Series) -> float:
    """Ridge of a baseline's past errors on the price features, applied to `now`.

    The penalty is chosen by expanding-window validation inside the training rows. A
    missing price is filled with its training mean, which after scaling means "no news".
    """
    rows = past.index.intersection(residual.index)
    model = make_pipeline(
        SimpleImputer(keep_empty_features=True),
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5)),
    )
    model.fit(past.loc[rows].to_numpy(), residual.loc[rows].to_numpy())
    return float(model.predict(now.to_numpy().reshape(1, -1))[0])


def arima_plus_ridge(history: pd.Series, past: pd.DataFrame, now: pd.Series) -> float:
    """ARIMA forecast plus a ridge model of ARIMA's one-step errors."""
    fitted = baselines.fit_airline(history.to_numpy(dtype=float))
    residual = pd.Series(fitted.resid, index=history.index)
    return float(fitted.forecast(1)[0]) + _ridge_correction(residual, past, now)


def seasonal_plus_ridge(history: pd.Series, past: pd.DataFrame, now: pd.Series) -> float:
    """Seasonal mean plus a ridge model of what the seasonal mean misses."""
    months = pd.PeriodIndex(history.index).month
    residual = history - history.groupby(months).transform("mean")
    return baselines.seasonal_mean(history) + _ridge_correction(residual, past, now)


def arimax_vegetables(history: pd.Series, past: pd.DataFrame, now: pd.Series) -> float:
    """ARIMA with tomato, onion and potato price changes as regressors."""
    columns = list(VEGETABLES)
    exog = past[columns].fillna(0.0)  # a missing price counts as no change
    fitted = baselines.fit_airline(history.loc[exog.index].to_numpy(dtype=float), exog.to_numpy())
    exog_now = now[columns].fillna(0.0).to_numpy(dtype=float).reshape(1, -1)
    return float(fitted.forecast(1, exog=exog_now)[0])


MODELS: dict[str, FeatureForecaster] = {
    "arimax_vegetables": arimax_vegetables,
    "seasonal_plus_ridge": seasonal_plus_ridge,
    "arima_plus_ridge": arima_plus_ridge,
}
