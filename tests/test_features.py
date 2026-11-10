import numpy as np
import pandas as pd

from nowcaster import backtest, features, models


def _price_index(months: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.period_range("2012-01", periods=months, freq="M")
    steps = rng.normal(0, 0.05, (months, len(features.COMMODITIES)))
    return pd.DataFrame(steps.cumsum(axis=0), index=index, columns=list(features.COMMODITIES))


def test_build_aligns_same_month_and_lagged_changes() -> None:
    prices = _price_index()
    table = features.build(prices)
    change = prices["Onions"].diff() * 100
    month = pd.Period("2014-06")
    assert table.loc[month, "Onions"] == change[month]
    assert table.loc[month, "Onions lag1"] == change[month - 1]
    assert table.shape[1] == 2 * len(features.COMMODITIES)


def test_build_starts_after_the_last_commodity_starts() -> None:
    prices = _price_index()
    prices.loc[:"2012-06", "Tomatoes"] = np.nan  # tomato prices begin in July 2012
    table = features.build(prices)
    assert table.index[0] == pd.Period("2012-09")  # first month with a lagged change
    assert not table.iloc[0].isna().any()


def test_run_with_features_never_passes_a_later_month() -> None:
    table = features.build(_price_index())
    series = pd.Series(np.arange(59.0), index=pd.period_range("2012-02", periods=59, freq="M"))
    seen: list[tuple[pd.Period, pd.Period, pd.Period]] = []

    def spy(history: pd.Series, past: pd.DataFrame, now: pd.Series) -> float:
        seen.append((history.index[-1], past.index[-1], now.name))
        return 0.0

    result = backtest.run_with_features(series, table, {"spy": spy}, pd.Period("2016-01"))
    assert len(result) == 12
    for target, (last_history, last_past, now) in zip(result.index, seen, strict=True):
        assert last_history == target - 1 and last_past == target - 1 and now == target


def test_arimax_uses_the_same_month_vegetable_prices() -> None:
    # The target is 0.5 x the tomato change plus small noise, so a model that reads the
    # target month's tomato price must land close to it.
    rng = np.random.default_rng(1)
    table = features.build(_price_index(months=100)).fillna(0.0)
    series = 0.5 * table["Tomatoes"] + rng.normal(0, 0.05, len(table))
    target = table.index[-1]
    forecast = models.arimax_vegetables(
        series[series.index < target], table[table.index < target], table.loc[target]
    )
    assert abs(forecast - 0.5 * table.loc[target, "Tomatoes"]) < 0.3


def test_ridge_models_run_with_missing_prices() -> None:
    rng = np.random.default_rng(2)
    table = features.build(_price_index(months=100))
    table.iloc[40:45] = np.nan  # a hole like the one in the WFP file
    series = pd.Series(rng.normal(0, 1, len(table)), index=table.index)
    target = table.index[-1]
    now = table.loc[target].copy()
    now["Tomatoes"] = np.nan
    for model in (models.arima_plus_ridge, models.seasonal_plus_ridge):
        value = model(series[series.index < target], table[table.index < target], now)
        assert np.isfinite(value)
