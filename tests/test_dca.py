from pathlib import Path

import pandas as pd
import pytest

from nowcaster import dca

PAGE = (Path(__file__).parent / "fixtures" / "dca_dailyprices.html").read_text(encoding="utf-8")


def test_parse_reads_both_sections_of_a_real_page() -> None:
    table = dca.parse(PAGE)
    retail = table[table["price_type"] == "retail"]
    wholesale = table[table["price_type"] == "wholesale"]
    assert retail["commodity"].nunique() == 42
    assert wholesale["commodity"].nunique() == 24
    assert table["date"].nunique() == 9
    assert table["date"].max() == pd.Timestamp("2026-11-09")

    def price(frame: pd.DataFrame, commodity: str, day: str) -> float:
        row = frame[(frame["commodity"] == commodity) & (frame["date"] == day)]
        return float(row["price"].iloc[0])

    assert price(retail, "Onion", "2026-11-09") == 54.32
    assert price(retail, "Onion", "2025-11-09") == 25.17
    assert price(wholesale, "Tomato", "2026-11-09") == 3087.71


def test_parse_drops_zero_prices() -> None:
    table = dca.parse(PAGE)
    broken = table[(table["commodity"] == "Broken Rice") & (table["price_type"] == "retail")]
    assert pd.Timestamp("2025-11-09") not in set(broken["date"])  # printed as 0 on the page
    assert (table["price"] > 0).all()


def test_parse_fails_loudly_on_a_changed_layout() -> None:
    with pytest.raises(ValueError, match="layout"):
        dca.parse("<html><body>maintenance</body></html>")


def test_load_keeps_the_later_snapshot(tmp_path: Path) -> None:
    (tmp_path / "Dailyprices_2026-11-09.html").write_text(PAGE, encoding="utf-8")
    revised = PAGE.replace(">54.32<", ">55.00<")
    (tmp_path / "Dailyprices_2026-11-10.html").write_text(revised, encoding="utf-8")
    table = dca.load(tmp_path)
    onion = table[(table["commodity"] == "Onion") & (table["price_type"] == "retail")
                  & (table["date"] == "2026-11-09")]
    assert onion["price"].tolist() == [55.0]
