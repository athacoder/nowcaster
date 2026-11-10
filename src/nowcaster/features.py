"""Price features for the end-of-month forecast.

The row for target month t holds each commodity's national price change in month t and in
month t-1. Prices for month t are published before the CPI for month t, so using them is
not a leak; using month t+1 would be.
"""

import pandas as pd

# Fixed before any model was scored: the commodities whose WFP history starts by August
# 2012, so every backtest month has at least 40 months to train on.
COMMODITIES = (
    "Rice", "Wheat", "Wheat flour", "Lentils (masur)", "Sugar", "Milk (pasteurized)",
    "Oil (groundnut)", "Oil (mustard)", "Oil (palm)", "Oil (sunflower)", "Salt (iodised)",
    "Potatoes", "Onions", "Tomatoes",
)
VEGETABLES = ("Tomatoes", "Onions", "Potatoes")


def build(price_index: pd.DataFrame) -> pd.DataFrame:
    """`price_index` is the log national index from `wfp.national_index`.

    Returns percent changes, one row per month, columns `<commodity>` (same month) and
    `<commodity> lag1`. A month with no price, or the month after one, stays missing.
    Rows before every commodity has started are dropped.
    """
    change = price_index[list(COMMODITIES)].diff() * 100
    lagged = change.shift(1).add_suffix(" lag1")
    table = pd.concat([change, lagged], axis=1)
    started = change.apply(lambda column: column.first_valid_index()).max()
    return pd.DataFrame(table.loc[table.index > started])
