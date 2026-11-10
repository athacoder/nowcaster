import numpy as np
import pandas as pd
import pytest

from nowcaster import backtest, baselines


def _series(values: list[float], start: str = "2011-02") -> pd.Series:
    return pd.Series(values, index=pd.period_range(start, periods=len(values), freq="M"))


def test_run_never_shows_a_model_the_target_month_or_later() -> None:
    series = _series([float(i) for i in range(30)])
    seen: list[tuple[pd.Period, pd.Period]] = []

    def spy(history: pd.Series) -> float:
        seen.append((history.index[-1], history.index[-1] + 1))
        return float(history.iloc[-1])

    first = pd.Period("2013-01")
    result = backtest.run(series, {"spy": spy}, first)
    assert result.index[0] == first and len(result) == 7
    # Each call's history ended exactly one month before its target.
    assert [target for _, target in seen] == list(result.index)
    assert (result["spy"] == result["actual"] - 1).all()


def test_seasonal_naive_and_seasonal_mean() -> None:
    history = _series([float(i) for i in range(1, 25)], start="2020-01")  # 2020 and 2021
    assert baselines.seasonal_naive(history) == 13.0  # January 2021
    assert baselines.seasonal_mean(history) == 7.0  # mean of January 2020 and 2021


def test_arima_learns_a_seasonal_pattern() -> None:
    rng = np.random.default_rng(0)
    pattern = np.array([2.0, 1.0, 0.0, -1.0, -2.0, 0.0, 3.0, 1.0, -1.0, 0.0, 1.0, -0.5])
    history = _series(list(np.tile(pattern, 8) + rng.normal(0, 0.1, 96)), start="2012-01")
    assert baselines.arima(history) == pytest.approx(pattern[0], abs=0.3)


def test_scores() -> None:
    forecasts = pd.DataFrame({"actual": [1.0, 2.0], "m": [2.0, 0.0]})
    table = backtest.scores(forecasts)
    assert table.loc["m", "mae"] == 1.5
    assert table.loc["m", "rmse"] == pytest.approx(np.sqrt(2.5))


def test_diebold_mariano() -> None:
    rng = np.random.default_rng(1)
    actual = pd.Series(rng.normal(size=200))
    good = actual + rng.normal(0, 0.1, 200)
    bad = actual + rng.normal(0, 1.0, 200)
    statistic, p_value = backtest.diebold_mariano(actual, good, bad)
    assert statistic < 0 and p_value < 0.001
    assert backtest.diebold_mariano(actual, good, good) == (0.0, 1.0)
    # Two equally noisy forecasters: the test should not reject.
    other = actual + rng.normal(0, 1.0, 200)
    assert backtest.diebold_mariano(actual, bad, other)[1] > 0.05
