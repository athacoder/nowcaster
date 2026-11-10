"""CPI from the MoSPI open API (api.mospi.gov.in): download raw pages, load tidy tables.

Two series are needed because the 2024-base series only starts in January 2025:
- base 2012: Food and Beverages group and its subgroups, January 2011 to December 2025;
- base 2024: Food and beverages division down to items, January 2025 onwards.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from nowcaster.http import PoliteClient

API = "https://api.mospi.gov.in/api/cpi"
PAGE_SIZE = 100  # the API refuses a larger limit

# name -> (endpoint, query, first year). All India, Combined sector only.
QUERIES: dict[str, tuple[str, dict[str, str | int], int]] = {
    "cpi2012_back": (
        "getCPIIndex",
        {"base_year": "2012", "series": "Back", "state_code": 99, "sector_code": 3,
         "group_code": "0,1,7"},
        2011,
    ),
    "cpi2012_current": (
        "getCPIIndex",
        {"base_year": "2012", "series": "Current", "state_code": 99, "sector_code": 3,
         "group_code": "0,1,7"},
        2013,
    ),
    "cpi2024_back": ("getCPIData", {"base_year": "2024", "series": "Back", "sector_code": 3}, 2013),
    "cpi2024_current": (
        "getCPIData",
        {"base_year": "2024", "series": "Current", "state_code": 1, "sector_code": 3,
         "division_code": "0,1"},
        2025,
    ),
}
LAST_YEAR = {"cpi2012_back": 2013, "cpi2012_current": 2025, "cpi2024_back": 2024}
MAX_PASSES = 3

Record = dict[str, Any]


def _one_pass(
    client: PoliteClient, url: str, query: dict[str, str | int]
) -> tuple[list[Record], int]:
    rows: list[Record] = []
    page, total_pages, total = 1, 1, 0
    while page <= total_pages:
        params = {**query, "limit": PAGE_SIZE, "page": page, "Format": "JSON"}
        body = client.get(url, params=params).json()
        if "data" not in body:
            raise RuntimeError(f"{url} {query} page {page}: {body}")
        meta = body.get("meta_data", {})
        total_pages, total = int(meta.get("totalPages", 1)), int(meta.get("totalRecords", 0))
        rows.extend(body["data"])
        page += 1
    return rows, total


def fetch_complete(client: PoliteClient, url: str, query: dict[str, str | int]) -> list[Record]:
    """Every record of one query, complete and without repeats.

    The API's page order is not stable: a row can show up on two pages while another row
    is skipped. So pages are read again, up to MAX_PASSES times, until the number of
    distinct rows equals the total the API reports.
    """
    seen: dict[str, Record] = {}
    for _ in range(MAX_PASSES):
        rows, total = _one_pass(client, url, query)
        for row in rows:
            seen[json.dumps(row, sort_keys=True)] = row
        if len(seen) == total:
            return list(seen.values())
    raise RuntimeError(f"{url} {query}: {len(seen)} distinct rows after {MAX_PASSES} passes, "
                       f"API reports {total}")


# getCPIIndex (base 2012) hides rows at page boundaries on every pass, so its requests
# are cut into half-years, which fit on one page (15 series x 6 months).
HALF_YEARS = ("1,2,3,4,5,6", "7,8,9,10,11,12")


def fetch(raw_dir: Path, client: PoliteClient, this_year: int) -> dict[str, int]:
    """Download every query, one file per year. Returns records per query."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, (endpoint, query, first_year) in QUERIES.items():
        counts[name] = 0
        chunks: tuple[dict[str, str | int], ...] = ({},)
        if endpoint == "getCPIIndex":
            chunks = tuple({"month_code": months} for months in HALF_YEARS)
        for year in range(first_year, LAST_YEAR.get(name, this_year) + 1):
            rows: list[Record] = []
            for chunk in chunks:
                params = {**query, "year": year, **chunk}
                rows += fetch_complete(client, f"{API}/{endpoint}", params)
            (raw_dir / f"{name}_{year}.json").write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8"
            )
            counts[name] += len(rows)
    return counts


def _records(raw_dir: Path, name: str) -> list[Record]:
    files = sorted(raw_dir.glob(f"{name}_[0-9][0-9][0-9][0-9].json"))
    if not files:
        raise FileNotFoundError(f"no {name} files in {raw_dir}; run `nowcaster fetch mospi`")
    rows: list[Record] = []
    for path in files:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _period(frame: pd.DataFrame) -> pd.Series:
    text = frame["year"].astype(str) + " " + frame["month"].astype(str)
    return pd.to_datetime(text, format="%Y %B").dt.to_period("M")


def _finish(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frame["index"] = pd.to_numeric(frame["index"], errors="coerce")
    frame = frame.dropna(subset=["index"])
    if frame.duplicated(keys).any():
        raise ValueError(f"duplicate rows for {keys}")
    return frame.sort_values(keys).reset_index(drop=True)


def load_2012(raw_dir: Path) -> pd.DataFrame:
    """Columns: period, group, subgroup, index. All India, Combined."""
    back = pd.DataFrame(_records(raw_dir, "cpi2012_back"))
    current = pd.DataFrame(_records(raw_dir, "cpi2012_current"))
    # 2013 is in both; the current series wins.
    back = back[back["year"].astype(int) < current["year"].astype(int).min()]
    frame = pd.concat([back, current])
    frame = frame[(frame["state"] == "All India") & (frame["sector"] == "Combined")].copy()
    frame["period"] = _period(frame)
    return _finish(frame[["period", "group", "subgroup", "index"]], ["period", "group", "subgroup"])


LEVELS = ["item", "sub_class", "class", "group", "division"]  # most specific first


def load_2024(raw_dir: Path) -> pd.DataFrame:
    """Columns: period, level, name, code, division, index. All India, Combined.

    Each row of the API is one node of the hierarchy; its level is the most specific
    field that is filled in.
    """
    frame = pd.DataFrame(_records(raw_dir, "cpi2024_current"))
    frame = frame[(frame["state"] == "All India") & (frame["sector"] == "Combined")].copy()
    frame["period"] = _period(frame)
    frame["level"] = "division"
    frame["name"] = frame["division"]
    for level in reversed(LEVELS[:-1]):
        filled = frame[level].notna()
        frame.loc[filled, "level"] = level
        frame.loc[filled, "name"] = frame.loc[filled, level]
    out = frame[["period", "level", "name", "code", "division", "index"]]
    return _finish(out, ["period", "level", "name", "code"])
