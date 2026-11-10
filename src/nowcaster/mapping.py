"""The hand-made table that ties price-source commodities to CPI items."""

from pathlib import Path

import pandas as pd

MAP_PATH = Path("data/reference/commodity_map.csv")
ITEMS_PATH = Path("data/reference/cpi2024_food_items.csv")
MATCHES = {"exact", "broad", "none"}


def load(path: Path = MAP_PATH) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table["cpi2024_item_code"] = pd.to_numeric(table["cpi2024_item_code"]).astype("Int64")
    return table


def problems(table: pd.DataFrame, items: pd.DataFrame) -> list[str]:
    """Everything wrong with the mapping, as plain sentences. Empty means valid.

    `items` is the CPI 2024 food item list with columns item_code and item.
    """
    found: list[str] = []
    for column in ("dca_commodity", "wfp_commodity"):
        names = table.loc[table[column] != "", column]
        found += [f"{column} {name!r} appears twice" for name in names[names.duplicated()]]
    found += [f"bad match value {m!r}" for m in set(table["match"]) - MATCHES]
    official = dict(zip(items["item_code"], items["item"], strict=True))
    for row in table.to_dict("records"):
        name, code, match = row["dca_commodity"], row["cpi2024_item_code"], row["match"]
        mapped = not pd.isna(code)
        if (match == "none") == mapped:
            found.append(f"{name}: match is {match!r} but item code is "
                         f"{'set' if mapped else 'empty'}")
        elif mapped and official.get(int(code)) != row["cpi2024_item"]:
            found.append(f"{name}: item {code} is not {row['cpi2024_item']!r} "
                         "in the CPI item list")
    return found
