"""Live predictions: one JSON file per month, made before the CPI release, scored after."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcaster import baselines, conformal, models
from nowcaster.features import VEGETABLES

PRIMARY = "arimax_vegetables"  # best on the backtest; chosen before any live month
REFERENCE = "arima"


def make(
    series: pd.Series, table: pd.DataFrame, food_2024: pd.Series, abs_errors: pd.DataFrame
) -> dict[str, Any]:
    """Forecast the first month after `series` ends.

    `food_2024` is the base-2024 Food and beverages index; `abs_errors` holds each model's
    absolute backtest errors by month. Refuses rather than guesses when the target month's
    vegetable prices are missing, or when the backtest does not reach the last released
    month.
    """
    month = series.index[-1] + 1
    if month not in table.index or table.loc[month, list(VEGETABLES)].isna().any():
        raise ValueError(f"no vegetable prices for {month}; fetch the price file again")
    if abs_errors.index[-1] != series.index[-1]:
        raise ValueError(f"backtest ends {abs_errors.index[-1]}, CPI ends {series.index[-1]}; "
                         "run `nowcaster backtest` first")
    past = table[table.index < month]
    point = {
        PRIMARY: models.arimax_vegetables(series, past, table.loc[month]),
        REFERENCE: baselines.arima(series),
    }
    result: dict[str, Any] = {"month": str(month), "target": "food_mom_pct", "models": {}}
    for name, forecast in point.items():
        errors = abs_errors[name].to_numpy(dtype=float)
        entry: dict[str, Any] = {"mom": round(forecast, 3)}
        for level in conformal.LEVELS:
            width = conformal.half_width(errors, level)
            entry[f"interval_{round(level * 100)}"] = [
                round(forecast - width, 3), round(forecast + width, 3)
            ]
        level = food_2024[month - 1] * math.exp(forecast / 100)
        entry["yoy"] = round((level / food_2024[month - 12] - 1) * 100, 2)
        result["models"][name] = entry
    result["backtest_months"] = len(abs_errors)
    return result


def record(predictions_dir: Path, series: pd.Series) -> pd.DataFrame:
    """One row per prediction and model, with the actual value once it is released.

    `verified_before_release` comes from `publication.csv` in the same folder: only a push
    to a public repository before the release proves a prediction came first. A month
    not listed there is left blank (not yet known).
    """
    publication_path = predictions_dir / "publication.csv"
    verified: dict[str, bool] = {}
    if publication_path.exists():
        published = pd.read_csv(publication_path, dtype=str)
        verified = dict(zip(published["month"], published["verified_before_release"] == "true",
                            strict=True))
    rows = []
    for path in sorted(predictions_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].json")):
        prediction = json.loads(path.read_text(encoding="utf-8"))
        month = pd.Period(prediction["month"], freq="M")
        actual = float(series.get(month, np.nan))
        for name, entry in prediction["models"].items():
            low80, high80 = entry["interval_80"]
            low95, high95 = entry["interval_95"]
            rows.append({
                "month": str(month), "made_at": prediction["made_at"], "model": name,
                "forecast": entry["mom"], "actual": actual,
                "abs_error": abs(actual - entry["mom"]),
                "inside_80": low80 <= actual <= high80 if not np.isnan(actual) else None,
                "inside_95": low95 <= actual <= high95 if not np.isnan(actual) else None,
                "verified_before_release": verified.get(str(month)),
            })
    return pd.DataFrame(rows)
