"""Monthly retail food prices for India from WFP (HDX, CC BY 3.0 IGO).

WFP republishes the Department of Consumer Affairs prices as monthly averages per market.
The set of markets changes over time (18 cities in 1994, about 160 centres in 2022, five
zone averages from August 2023), so a plain mean across markets jumps when the set
changes. `national_index` chains month-on-month changes over markets present in both
months instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nowcaster.http import PoliteClient

PACKAGE_URL = "https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-india"


def fetch(raw_dir: Path, client: PoliteClient) -> Path:
    """Download the current CSV and the dataset record that states its licence and date."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    package = client.get(PACKAGE_URL)
    (raw_dir / "hdx_package.json").write_bytes(package.content)
    resources = package.json()["result"]["resources"]
    url = next(r["url"] for r in resources if r["url"].endswith("wfp_food_prices_ind.csv"))
    path = raw_dir / "wfp_food_prices_ind.csv"
    path.write_bytes(client.get(url).content)
    return path


def load(path: Path) -> pd.DataFrame:
    """Columns: period, market, commodity, unit, price. Retail prices only."""
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[(frame["pricetype"] == "Retail") & (frame["price"] > 0)].copy()
    frame["period"] = frame["date"].dt.to_period("M")
    keys = ["period", "market", "commodity"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate market-commodity-month rows")
    frame = frame[[*keys, "unit", "price"]].sort_values(keys).reset_index(drop=True)
    return frame[~copied_months(frame)].reset_index(drop=True)


def copied_months(prices: pd.DataFrame) -> pd.Series:
    """Rows of commodity-months that repeat the previous month exactly in every market.

    WFP fills some months by copying the month before (all 21 commodities in all five
    zones in 2024-11, 2024-12, 2026-04, 2026-07 and 2026-08). Those are not observations,
    and keeping them makes the next real month carry the whole multi-month change.
    """
    wide = prices.pivot(index=["commodity", "period"], columns="market", values="price")
    previous = wide.groupby(level="commodity").shift(1)
    same_markets = wide.notna().eq(previous.notna()).all(axis=1)
    observed = wide.notna().sum(axis=1)
    copied = same_markets & (observed >= 2) & (wide == previous).sum(axis=1).eq(observed)
    # Only a copy of the month right before counts, not of a month before a gap.
    periods = wide.index.get_level_values("period")
    previous_period = pd.Series(periods, index=wide.index).groupby(level="commodity").shift(1)
    copied &= previous_period.eq(pd.Series(periods - 1, index=wide.index))
    flagged = set(copied[copied].index)
    return pd.Series(
        [(c, p) in flagged for c, p in zip(prices["commodity"], prices["period"], strict=True)],
        index=prices.index,
    )


def national_index(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chained national log price index per commodity.

    Returns (index, breaks). `index` has one column per commodity, rows are months, values
    are log price relative to the commodity's first month. `breaks` lists the
    (period, commodity) links where no market was present in both months, so the change
    was taken from the ratio of cross-market means; treat those months with care.
    """
    columns: dict[str, pd.Series] = {}
    breaks: list[dict[str, object]] = []
    for commodity, group in prices.groupby("commodity"):
        wide = pd.DataFrame(np.log(group.pivot(index="period", columns="market", values="price")))
        # Rows are the months that have any price, so a change can span a gap; the gap
        # months themselves stay missing in the result.
        matched = wide.diff().mean(axis=1)  # mean over markets present in both months
        level = pd.Series(np.log(np.exp(wide).mean(axis=1)))
        unmatched = matched.isna()
        unmatched.iloc[0] = False
        change = matched.where(~unmatched, level.diff())
        for period in change.index[unmatched]:
            breaks.append({"period": period, "commodity": commodity})
        change.iloc[0] = 0.0
        columns[str(commodity)] = change.cumsum()
    index = pd.DataFrame(columns)
    index = index.reindex(pd.period_range(index.index.min(), index.index.max(), freq="M"))
    return index, pd.DataFrame(breaks, columns=["period", "commodity"])
