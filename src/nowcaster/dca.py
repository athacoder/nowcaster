"""Department of Consumer Affairs daily all-India average prices.

Only the public daily page is used. It shows today's price and eight look-back dates
(1-5 days, 1 week, 1 month, 1 year). The full history sits behind a CAPTCHA and is not
fetched; saving this page daily builds the series forward.
"""

import html as html_lib
import re
from pathlib import Path

import pandas as pd

from nowcaster.http import PoliteClient

URL = "https://fcainfoweb.nic.in/Reports/DB/Dailyprices.aspx"

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_DATE = re.compile(r"\d{2}/\d{2}/\d{4}")
_NUMBER = re.compile(r"\d+(\.\d+)?")
_WHOLESALE_TITLE = "Daily Average Wholesale Prices"


def _cells(row: str) -> list[str]:
    return [html_lib.unescape(_TAG.sub("", c)).strip() for c in _CELL.findall(row)]


def _parse_section(section: str, price_type: str) -> list[dict[str, object]]:
    dates: list[pd.Timestamp] = []
    out: list[dict[str, object]] = []
    for row in _ROW.findall(section):
        cells = _cells(row)
        if not dates:
            found = [c for c in cells if _DATE.fullmatch(c)]
            if len(found) >= 2:
                dates = [pd.to_datetime(c, format="%d/%m/%Y") for c in found]
            continue
        if len(cells) != len(dates) + 2:
            continue
        commodity, unit, values = cells[0], cells[1], cells[2:]
        if not all(_NUMBER.fullmatch(v) for v in values):
            raise ValueError(f"unexpected price cell for {commodity}: {values}")
        for date, value in zip(dates, values, strict=True):
            price = float(value)
            if price > 0:  # the page prints 0 when a date has no price
                out.append(
                    {"date": date, "commodity": commodity, "unit": unit,
                     "price_type": price_type, "price": price}
                )
    if not out:
        raise ValueError(f"no {price_type} prices found; the page layout may have changed")
    return out


def parse(page: str) -> pd.DataFrame:
    """Columns: date, commodity, unit, price_type (retail or wholesale), price (rupees)."""
    split = page.find(_WHOLESALE_TITLE)
    if split < 0:
        raise ValueError("wholesale section not found; the page layout may have changed")
    rows = _parse_section(page[:split], "retail") + _parse_section(page[split:], "wholesale")
    return pd.DataFrame(rows)


def fetch(raw_dir: Path, client: PoliteClient) -> Path:
    """Save today's page under the report date it carries. Returns the file written."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    page = client.get(URL).text
    report_date = parse(page)["date"].max()
    path = raw_dir / f"Dailyprices_{report_date:%Y-%m-%d}.html"
    path.write_text(page, encoding="utf-8")
    return path


def load(raw_dir: Path) -> pd.DataFrame:
    """All saved pages as one table. A later page wins where two pages give one date."""
    frames = []
    for path in sorted(raw_dir.glob("Dailyprices_*.html")):
        frame = parse(path.read_text(encoding="utf-8"))
        frame["snapshot"] = path.stem.removeprefix("Dailyprices_")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no pages in {raw_dir}; run `nowcaster fetch dca`")
    keys = ["date", "commodity", "price_type"]
    table = pd.concat(frames).sort_values([*keys, "snapshot"])
    return table.drop_duplicates(keys, keep="last").drop(columns="snapshot").reset_index(drop=True)
