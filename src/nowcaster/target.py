"""The forecast target: month-on-month change in the all-India Food and Beverages index.

Two official series cover it: base 2012 up to December 2025 and base 2024 from January
2025. The target takes each month's change from the series that was the official one at
the time, so no change is ever computed across the two bases.

Why not join the index levels instead: MoSPI links its General index by scaling the old
series with the ratio of the 2025 averages. Done for food, that puts a false drop at the
join (January 2025: -3.65% instead of the published -2.46%), because the two series move
differently inside 2025.
"""

import numpy as np
import pandas as pd

# CPI for January 2026 (released 12 February 2026) was the first on base 2024.
FIRST_2024_BASE_MONTH = pd.Period("2026-01", freq="M")
FOOD_2012 = ("Food and Beverages", "Food and Beverages-Overall")
FOOD_2024 = "Food and beverages"


def series_2012(cpi2012: pd.DataFrame, group: str, subgroup: str) -> pd.Series:
    rows = cpi2012[(cpi2012["group"] == group) & (cpi2012["subgroup"] == subgroup)]
    return rows.set_index("period")["index"]


def division_2024(cpi2024: pd.DataFrame, name: str) -> pd.Series:
    rows = cpi2024[(cpi2024["level"] == "division") & (cpi2024["name"] == name)]
    return rows.set_index("period")["index"]


def mom(index: pd.Series) -> pd.Series:
    """Month-on-month change in percent (log difference x 100)."""
    return pd.Series(np.log(index)).diff().mul(100).dropna().rename("mom")


def joined_mom(old: pd.Series, new: pd.Series) -> pd.Series:
    """Changes of `old` (base 2012) before FIRST_2024_BASE_MONTH, of `new` from it on."""
    old_changes, new_changes = mom(old), mom(new)
    joined: pd.Series = pd.concat([
        old_changes[old_changes.index < FIRST_2024_BASE_MONTH],
        new_changes[new_changes.index >= FIRST_2024_BASE_MONTH],
    ])
    expected = pd.period_range(joined.index[0], joined.index[-1], freq="M")
    if not joined.index.equals(expected):
        raise ValueError("target has missing or repeated months")
    return joined


def food_mom(cpi2012: pd.DataFrame, cpi2024: pd.DataFrame) -> pd.Series:
    return joined_mom(series_2012(cpi2012, *FOOD_2012), division_2024(cpi2024, FOOD_2024))
