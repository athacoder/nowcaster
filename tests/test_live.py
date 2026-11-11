import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcaster import features, live


def _inputs() -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(0)
    months = pd.period_range("2012-01", "2026-10", freq="M")
    prices = pd.DataFrame(
        rng.normal(0, 0.05, (len(months), len(features.COMMODITIES))).cumsum(axis=0),
        index=months, columns=list(features.COMMODITIES),
    )
    table = features.build(prices)
    series = pd.Series(rng.normal(0.4, 1.0, len(table)), index=table.index)
    series = series[series.index <= pd.Period("2026-09")]
    food = pd.Series(100.0, index=pd.period_range("2025-01", "2026-09", freq="M"))
    backtest_months = series.index[series.index >= pd.Period("2016-01")]
    errors = pd.DataFrame({live.PRIMARY: 0.5, live.REFERENCE: 0.7}, index=backtest_months)
    return series, table, food, errors


def test_make_forecasts_the_next_month_with_intervals_and_yoy() -> None:
    series, table, food, errors = _inputs()
    prediction = live.make(series, table, food, errors)
    assert prediction["month"] == "2026-10"
    entry = prediction["models"][live.PRIMARY]
    assert entry["interval_80"] == pytest.approx([entry["mom"] - 0.5, entry["mom"] + 0.5])
    # Index flat at 100, so year-on-year equals the month's own change.
    assert entry["yoy"] == pytest.approx((math.exp(entry["mom"] / 100) - 1) * 100, abs=0.01)


def test_make_refuses_without_vegetable_prices() -> None:
    series, table, food, errors = _inputs()
    table.loc[pd.Period("2026-10"), "Onions"] = np.nan
    with pytest.raises(ValueError, match="no vegetable prices"):
        live.make(series, table, food, errors)


def test_make_refuses_a_stale_backtest() -> None:
    series, table, food, errors = _inputs()
    with pytest.raises(ValueError, match="run `nowcaster backtest`"):
        live.make(series, table, food, errors.iloc[:-1])


def test_record_scores_only_released_months(tmp_path: Path) -> None:
    for month, forecast in (("2026-09", 1.0), ("2026-10", 0.5)):
        prediction = {
            "month": month, "made_at": "x",
            "models": {"m": {"mom": forecast, "interval_80": [forecast - 0.5, forecast + 0.5],
                             "interval_95": [forecast - 1.0, forecast + 1.0]}},
        }
        (tmp_path / f"{month}.json").write_text(json.dumps(prediction), encoding="utf-8")
    series = pd.Series([1.2], index=pd.PeriodIndex(["2026-09"], freq="M"))
    table = live.record(tmp_path, series).set_index("month")
    assert table.loc["2026-09", "abs_error"] == pytest.approx(0.2)
    assert table.loc["2026-09", "inside_80"]
    assert np.isnan(table.loc["2026-10", "actual"])
