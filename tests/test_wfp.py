import numpy as np
import pandas as pd
import pytest

from nowcaster import wfp


def _prices(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["period", "market", "price"])
    frame["period"] = pd.PeriodIndex(frame["period"], freq="M")
    frame["commodity"] = "Onions"
    return frame


def test_index_ignores_a_market_joining_at_a_different_price_level() -> None:
    # Delhi rises 10% a month. Mumbai joins in month 2 at five times the price and also
    # rises 10%. A plain mean would jump; the chained index must not.
    prices = _prices([
        ("2020-01", "Delhi", 10.0),
        ("2020-02", "Delhi", 11.0), ("2020-02", "Mumbai", 50.0),
        ("2020-03", "Delhi", 12.1), ("2020-03", "Mumbai", 55.0),
    ])
    index, breaks = wfp.national_index(prices)
    assert np.exp(index["Onions"]).tolist() == pytest.approx([1.0, 1.1, 1.21])
    assert breaks.empty


def test_index_flags_a_month_with_no_common_market() -> None:
    prices = _prices([
        ("2023-07", "Delhi", 10.0), ("2023-08", "North Zone", 12.0),
        ("2023-09", "North Zone", 12.0),
    ])
    index, breaks = wfp.national_index(prices)
    assert breaks["period"].astype(str).tolist() == ["2023-08"]
    assert np.exp(index["Onions"]).tolist() == pytest.approx([1.0, 1.2, 1.2])


def test_index_continues_after_a_missing_month() -> None:
    prices = _prices([("2020-01", "Delhi", 10.0), ("2020-03", "Delhi", 20.0)])
    index, _ = wfp.national_index(prices)
    assert index.index.astype(str).tolist() == ["2020-01", "2020-02", "2020-03"]
    assert np.isnan(index["Onions"].iloc[1])
    assert np.exp(index["Onions"].iloc[2]) == pytest.approx(2.0)
