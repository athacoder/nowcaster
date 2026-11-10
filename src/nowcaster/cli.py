"""Nowcast India's monthly food CPI before the official release."""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from nowcaster import __version__, backtest, baselines, dca, mapping, mospi, target, wfp
from nowcaster.http import PoliteClient

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
FIRST_FORECAST = pd.Period("2016-01", freq="M")  # leaves five years to fit on


def _fetch(args: argparse.Namespace) -> int:
    client = PoliteClient(legacy_tls=args.source == "mospi")
    try:
        if args.source == "mospi":
            for name, n in mospi.fetch(RAW / "mospi", client, date.today().year).items():
                print(f"{name}: {n} records")
        elif args.source == "dca":
            print(f"saved {dca.fetch(RAW / 'dca', client)}")
        else:
            print(f"saved {wfp.fetch(RAW / 'wfp', client)}")
    finally:
        client.close()
    return 0


def _check(args: argparse.Namespace) -> int:
    """Validate the commodity map against the CPI item list and the saved DCA pages."""
    table = mapping.load()
    found = mapping.problems(table, pd.read_csv(mapping.ITEMS_PATH))
    on_page = set(dca.load(RAW / "dca")["commodity"])
    unmapped = on_page - set(table["dca_commodity"])
    found += [f"DCA commodity {c!r} is not in the map" for c in unmapped]
    for problem in found:
        print(problem)
    mapped = table[table["match"] != "none"]
    print(f"{len(mapped)} of {len(table)} DCA commodities mapped to "
          f"{mapped['cpi2024_item_code'].nunique()} CPI items; {len(found)} problems")
    return 1 if found else 0


def _backtest(args: argparse.Namespace) -> int:
    series = target.food_mom(mospi.load_2012(RAW / "mospi"), mospi.load_2024(RAW / "mospi"))
    forecasts = backtest.run(series, baselines.BASELINES, FIRST_FORECAST)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(PROCESSED / "backtest_baselines.csv")

    print(f"target: Food and Beverages, month-on-month % change, {series.index[0]} to "
          f"{series.index[-1]}")
    print(f"backtest: {len(forecasts)} one-step forecasts, {forecasts.index[0]} to "
          f"{forecasts.index[-1]}\n")
    table = backtest.scores(forecasts)
    reference = "seasonal_naive"
    for name in table.index:
        if name == reference:
            table.loc[name, ["dm_stat", "dm_p"]] = [float("nan")] * 2
        else:
            table.loc[name, ["dm_stat", "dm_p"]] = backtest.diebold_mariano(
                forecasts["actual"], forecasts[name], forecasts[reference]
            )
    table["mae_vs_naive"] = table["mae"] / float(table["mae"][reference])
    shown = table.round(3).astype(object)
    shown["dm_p"] = table["dm_p"].map("{:.2g}".format)
    print(shown.to_string())
    print(f"\ndm_stat < 0: lower absolute error than {reference}. Units: percentage points.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nowcaster", description=__doc__)
    parser.add_argument("--version", action="version", version=f"nowcaster {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download a source into data/raw/")
    fetch.add_argument("source", choices=["mospi", "dca", "wfp"])
    fetch.set_defaults(func=_fetch)
    sub.add_parser("check", help=_check.__doc__).set_defaults(func=_check)
    sub.add_parser("backtest", help="backtest the baselines").set_defaults(func=_backtest)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
