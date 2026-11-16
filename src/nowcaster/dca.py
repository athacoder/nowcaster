"""Department of Consumer Affairs daily all-India average prices.

Only the public daily page is used. It shows today's price and eight look-back dates
(1-5 days, 1 week, 1 month, 1 year). The full history sits behind a CAPTCHA and is not
fetched; saving this page daily builds the series forward.
"""

import gzip
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


def fetch(pages_dir: Path, client: PoliteClient) -> Path:
    """Save today's page, gzipped, under the report date it carries. Returns the file.

    Kept raw so a parser fix can be replayed. The page also carries a visitor counter and
    form state that change on every request, so an existing file is replaced only when the
    prices differ (a weekend figure revised on Monday, for example). Reruns then write
    nothing and make no commit.
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    page = client.get(URL).text
    prices = parse(page)
    path = pages_dir / f"Dailyprices_{prices['date'].max():%Y-%m-%d}.html.gz"
    if not (path.exists() and parse(_read(path)).equals(prices)):
        path.write_bytes(gzip.compress(page.encode("utf-8"), mtime=0))
    return path


def _read(path: Path) -> str:
    data = path.read_bytes()
    return (gzip.decompress(data) if path.suffix == ".gz" else data).decode("utf-8")


def load(pages_dir: Path) -> pd.DataFrame:
    """All saved pages as one table. A later page wins where two pages give one date."""
    frames = []
    for path in sorted(pages_dir.glob("Dailyprices_*.html*")):
        frame = parse(_read(path))
        frame["snapshot"] = path.name.removeprefix("Dailyprices_")[:10]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no pages in {pages_dir}; run `nowcaster fetch dca`")
    keys = ["date", "commodity", "price_type"]
    table = pd.concat(frames).sort_values([*keys, "snapshot"])
    return table.drop_duplicates(keys, keep="last").drop(columns="snapshot").reset_index(drop=True)


def write_table(pages_dir: Path, csv_path: Path) -> int:
    """Rebuild the tracked CSV from every saved page. Returns its row count."""
    table = load(pages_dir)
    table.to_csv(csv_path, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    return len(table)
