import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from nowcaster import mospi
from nowcaster.http import PoliteClient


class FakeClient(PoliteClient):
    """Serves canned passes; each pass is a list of pages."""

    def __init__(self, passes: list[list[list[dict[str, Any]]]], total: int) -> None:
        self.passes, self.total, self.calls = passes, total, 0

    def get(self, url: str, params: dict[str, str | int] | None = None) -> httpx.Response:
        assert params is not None
        pages = self.passes[min(self.calls // 2, len(self.passes) - 1)]
        self.calls += 1
        body = {
            "data": pages[int(params["page"]) - 1],
            "meta_data": {"totalPages": len(pages), "totalRecords": self.total},
        }
        return httpx.Response(200, json=body)


A, B, C, D = ({"id": i} for i in range(4))


def test_fetch_complete_recovers_rows_lost_to_unstable_paging() -> None:
    # Pass 1 repeats B and skips C; pass 2 is in a different order and has C.
    client = FakeClient([[[A, B], [B, D]], [[A, C], [B, D]]], total=4)
    rows = mospi.fetch_complete(client, "u", {})
    assert sorted(r["id"] for r in rows) == [0, 1, 2, 3]


def test_fetch_complete_fails_loudly_when_never_complete() -> None:
    client = FakeClient([[[A, B], [B, D]]], total=4)
    with pytest.raises(RuntimeError, match="3 distinct rows"):
        mospi.fetch_complete(client, "u", {})


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _row2012(year: int, month: str, index: str) -> dict[str, Any]:
    return {"year": year, "month": month, "state": "All India", "sector": "Combined",
            "group": "Food and Beverages", "subgroup": "Food and Beverages-Overall",
            "index": index}


def test_load_2012_prefers_current_where_back_overlaps(tmp_path: Path) -> None:
    _write(tmp_path / "cpi2012_back_2012.json",
           [_row2012(2012, "December", "105.0"), _row2012(2013, "January", "1.0")])
    _write(tmp_path / "cpi2012_current_2013.json", [_row2012(2013, "January", "105.9")])
    table = mospi.load_2012(tmp_path)
    assert table["period"].astype(str).tolist() == ["2012-12", "2013-01"]
    assert table["index"].tolist() == [105.0, 105.9]


def test_load_2024_names_each_row_by_its_most_specific_level(tmp_path: Path) -> None:
    base = {"year": "2026", "month": "September", "state": "All India", "sector": "Combined",
            "division": "Food and beverages", "group": None, "class": None,
            "sub_class": None, "item": None, "code": "01", "index": "111.77"}
    item = {**base, "group": "Food", "class": "Cereals and cereal products",
            "sub_class": "Cereals", "item": "Rice", "code": "01.1.1.1.1", "index": "104.2"}
    _write(tmp_path / "cpi2024_current_2026.json", [base, item])
    table = mospi.load_2024(tmp_path).set_index("name")
    assert table.loc["Food and beverages", "level"] == "division"
    assert table.loc["Rice", "level"] == "item"
    assert table.loc["Rice", "period"] == pd.Period("2026-09")
    assert table.loc["Rice", "index"] == 104.2
