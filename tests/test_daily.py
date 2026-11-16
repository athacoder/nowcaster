import gzip
from pathlib import Path

import httpx
import pandas as pd

from nowcaster import daily, dca, mospi, wfp
from nowcaster.http import PoliteClient

PAGE = (Path(__file__).parent / "fixtures" / "dca_dailyprices.html").read_text(encoding="utf-8")


def _predictions(folder: Path, months: list[str], scored: list[str]) -> None:
    for month in months:
        (folder / f"{month}.json").write_text("{}", encoding="utf-8")
    rows = [f"{month},x,m,0.5,{1.0 if month in scored else ''}" for month in months]
    (folder / "record.csv").write_text(
        "month,made_at,model,forecast,actual\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )


def test_target_month_skips_months_already_released(tmp_path: Path) -> None:
    _predictions(tmp_path, ["2026-10"], scored=[])
    out = {pd.Period("2026-11"), pd.Period("2026-12")}
    assert daily.target_month(tmp_path, lambda m: m in out) == pd.Period("2027-01")
    assert daily.target_month(tmp_path, lambda m: False) == pd.Period("2026-11")


def test_unscored_lists_months_without_an_actual(tmp_path: Path) -> None:
    _predictions(tmp_path, ["2026-10", "2026-11"], scored=["2026-10"])
    assert daily.unscored(tmp_path) == [pd.Period("2026-11")]
    (tmp_path / "record.csv").unlink()  # no record yet: everything is unscored
    assert daily.unscored(tmp_path) == [pd.Period("2026-10"), pd.Period("2026-11")]


def test_published_through_reads_the_hdx_date_range() -> None:
    package = {"result": {"dataset_date": "[1994-01-15T00:00:00 TO 2026-10-15T23:59:59]"}}
    assert wfp.published_through(package) == pd.Period("2026-10")


class _Client(PoliteClient):
    def __init__(self, body: object = None, text: str = "") -> None:
        self.body, self.text = body, text

    def get(self, url: str, params: dict[str, str | int] | None = None) -> httpx.Response:
        if self.text:
            return httpx.Response(200, text=self.text)
        return httpx.Response(200, json=self.body)


def test_released_is_true_only_when_the_api_returns_rows() -> None:
    month = pd.Period("2026-10")
    assert mospi.released(_Client({"data": [{"index": "112.78"}]}), month)
    assert not mospi.released(_Client({"data": [], "msg": "No Data Found"}), month)


def test_dca_fetch_rewrites_a_page_only_when_its_prices_change(tmp_path: Path) -> None:
    first = dca.fetch(tmp_path, _Client(text=PAGE))
    before = first.read_bytes()
    assert gzip.decompress(before).decode("utf-8") == PAGE
    # Same prices, different page chrome (the counter changes on every request).
    rerun = PAGE.replace("</body>", "<!-- visitor 22180930 --></body>")
    assert dca.fetch(tmp_path, _Client(text=rerun)).read_bytes() == before
    revised = PAGE.replace(">54.32<", ">55.00<")  # a revised onion price
    assert dca.fetch(tmp_path, _Client(text=revised)).read_bytes() != before
    rows = dca.write_table(tmp_path, tmp_path / "dca.csv")
    assert rows == len(dca.parse(PAGE))
    assert pd.read_csv(tmp_path / "dca.csv")["date"].max() == "2026-11-09"
