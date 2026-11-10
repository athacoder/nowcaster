from pathlib import Path

import pandas as pd

from nowcaster import dca, mapping

ROOT = Path(__file__).parent.parent
PAGE = (Path(__file__).parent / "fixtures" / "dca_dailyprices.html").read_text(encoding="utf-8")


def _items() -> pd.DataFrame:
    return pd.read_csv(ROOT / mapping.ITEMS_PATH)


def test_the_committed_mapping_is_valid() -> None:
    assert mapping.problems(mapping.load(ROOT / mapping.MAP_PATH), _items()) == []


def test_every_commodity_on_the_dca_page_has_a_row() -> None:
    table = mapping.load(ROOT / mapping.MAP_PATH)
    on_page = set(dca.parse(PAGE)["commodity"])
    assert on_page == set(table["dca_commodity"])


def test_problems_catches_a_wrong_item_and_a_duplicate() -> None:
    table = mapping.load(ROOT / mapping.MAP_PATH)
    table.loc[table["dca_commodity"] == "Onion", "cpi2024_item_code"] = 98  # that is Potato
    table = pd.concat([table, table[table["dca_commodity"] == "Rice"]])
    found = mapping.problems(table, _items())
    assert any("Onion" in p and "98" in p for p in found)
    assert any("'Rice' appears twice" in p for p in found)
