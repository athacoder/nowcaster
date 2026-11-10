from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcaster import mospi, target

RAW = Path("data/raw/mospi")


def _growing(start: str, months: int, level: float, rate: float) -> pd.Series:
    index = pd.period_range(start, periods=months, freq="M")
    return pd.Series(level * (1 + rate) ** np.arange(months), index=index)


def test_mom_is_log_change_in_percent() -> None:
    index = pd.Series([100.0, 101.0, 101.0], index=pd.period_range("2025-01", periods=3, freq="M"))
    changes = target.mom(index)
    assert changes.tolist() == pytest.approx([np.log(1.01) * 100, 0.0])
    assert changes.index[0] == pd.Period("2025-02")


def test_joined_mom_never_takes_a_change_across_the_two_bases() -> None:
    # Old series: level 200, +1% a month, to December 2025.
    # New series: level 100, +2% a month, from January 2025.
    old = _growing("2024-01", 24, 200.0, 0.01)
    new = _growing("2025-01", 15, 100.0, 0.02)
    joined = target.joined_mom(old, new)
    assert joined.index[0] == pd.Period("2024-02") and joined.index[-1] == pd.Period("2026-03")
    assert joined[:"2025-12"].to_numpy() == pytest.approx(np.log(1.01) * 100)
    assert joined["2026-01":].to_numpy() == pytest.approx(np.log(1.02) * 100)


def test_joined_mom_rejects_a_gap() -> None:
    old = _growing("2024-01", 20, 200.0, 0.01)  # stops in August 2025
    new = _growing("2025-01", 15, 100.0, 0.02)
    with pytest.raises(ValueError, match="missing"):
        target.joined_mom(old, new)


@pytest.mark.skipif(not list(RAW.glob("cpi2012_current_*.json")), reason="needs downloaded CPI")
def test_food_target_on_the_real_download() -> None:
    cpi2012, cpi2024 = mospi.load_2012(RAW), mospi.load_2024(RAW)
    changes = target.food_mom(cpi2012, cpi2024)
    assert changes.index[0] == pd.Period("2011-02")
    # January 2025 is the published base-2012 change, not a jump from joining levels.
    assert changes["2025-01"] == pytest.approx(-2.46, abs=0.01)
    # The loaders agree with MoSPI's own year-on-year figure for September 2026 (7.36%).
    food = target.division_2024(cpi2024, target.FOOD_2024)
    assert (food["2026-09"] / food["2025-09"] - 1) * 100 == pytest.approx(7.36, abs=0.01)
