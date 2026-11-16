"""Nowcast India's monthly food CPI before the official release."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd

from nowcaster import (
    __version__,
    backtest,
    baselines,
    conformal,
    daily,
    dca,
    features,
    live,
    mapping,
    models,
    mospi,
    target,
    wfp,
)
from nowcaster.http import PoliteClient

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PREDICTIONS = Path("predictions")
WFP_FILE = RAW / "wfp" / "wfp_food_prices_ind.csv"
DCA_PAGES = Path("data/collected/dca_pages")
DCA_TABLE = Path("data/collected/dca.csv")
FIRST_FORECAST = pd.Period("2016-01", freq="M")  # leaves five years to fit on


def _fetch(args: argparse.Namespace) -> int:
    client = PoliteClient(legacy_tls=args.source == "mospi")
    try:
        if args.source == "mospi":
            for name, n in mospi.fetch(RAW / "mospi", client, date.today().year).items():
                print(f"{name}: {n} records")
        elif args.source == "dca":
            print(f"saved {dca.fetch(DCA_PAGES, client)}")
            print(f"{DCA_TABLE}: {dca.write_table(DCA_PAGES, DCA_TABLE)} rows")
        else:
            print(f"saved {wfp.fetch(RAW / 'wfp', client)}")
    finally:
        client.close()
    return 0


def _check(args: argparse.Namespace) -> int:
    """Validate the commodity map against the CPI item list and the saved DCA pages."""
    table = mapping.load()
    found = mapping.problems(table, pd.read_csv(mapping.ITEMS_PATH))
    on_page = set(dca.load(DCA_PAGES)["commodity"])
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
    price_index, _ = wfp.national_index(wfp.load(WFP_FILE))
    table = features.build(price_index)
    forecasts = backtest.run(series, baselines.BASELINES, FIRST_FORECAST).join(
        backtest.run_with_features(series, table, models.MODELS, FIRST_FORECAST)
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(PROCESSED / "backtest.csv")

    print(f"target: Food and Beverages, month-on-month % change, {series.index[0]} to "
          f"{series.index[-1]}")
    print(f"backtest: {len(forecasts)} one-step forecasts, {forecasts.index[0]} to "
          f"{forecasts.index[-1]}\n")
    result = backtest.scores(forecasts)
    actual = forecasts["actual"]
    errors = forecasts.drop(columns="actual").sub(actual, axis=0).abs()
    for reference in ("seasonal_naive", "arima"):
        result[f"mae_vs_{reference}"] = result["mae"] / float(result["mae"][reference])
    for name in result.index:
        if name != "arima":
            statistic, p_value = backtest.diebold_mariano(
                actual, forecasts[name], forecasts["arima"]
            )
            result.loc[name, "dm_vs_arima"] = statistic
            result.loc[name, "p"] = p_value
            result.loc[name, "months_won"] = float((errors[name] < errors["arima"]).sum())
    shown = result.round(3).astype(object)
    shown["p"] = result["p"].map("{:.2g}".format)
    print(shown.to_string())
    print("\ndm_vs_arima < 0: lower absolute error than arima. Units: percentage points.")

    print(f"\nconformal intervals, built from earlier months' errors only "
          f"(first {conformal.MIN_HISTORY} months used to start):")
    for name in errors.columns:
        cells = []
        for level in conformal.LEVELS:
            share, width, months = conformal.coverage(errors[name], level)
            cells.append(f"{level:.0%}: covered {share:.1%}, width {width:.2f}")
        print(f"  {name:<20} {'; '.join(cells)} ({months} months)")

    # The live fallback rule (ARIMA when a month has no vegetable prices) was written after
    # these errors were seen, so its score is post-hoc and is printed apart from the table.
    has_prices = table[list(features.VEGETABLES)].notna().all(axis=1)
    has_prices = has_prices.reindex(forecasts.index, fill_value=False)
    fallback = forecasts[live.PRIMARY].where(has_prices, forecasts[live.REFERENCE])
    statistic, p_value = backtest.diebold_mariano(actual, fallback, forecasts["arima"])
    print(f"\npost-hoc, rule chosen after seeing these errors: {live.PRIMARY} with ARIMA in "
          f"the {int((~has_prices).sum())} months without vegetable prices: MAE "
          f"{(fallback - actual).abs().mean():.3f}, DM vs arima {statistic:.2f} "
          f"(p {p_value:.2g})")
    return 0


def _vintage() -> dict[str, str]:
    """What the prediction was made from, so it can be reproduced or audited."""
    package = json.loads((RAW / "wfp" / "hdx_package.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "wfp_last_modified": package["result"]["last_modified"],
        "wfp_sha256": hashlib.sha256(WFP_FILE.read_bytes()).hexdigest(),
        "code_commit": commit,
    }


def _predict(args: argparse.Namespace) -> int:
    cpi2012, cpi2024 = mospi.load_2012(RAW / "mospi"), mospi.load_2024(RAW / "mospi")
    series = target.food_mom(cpi2012, cpi2024)
    month = series.index[-1] + 1
    path = PREDICTIONS / f"{month}.json"
    if path.exists():
        print(f"{path} exists; a prediction is never overwritten")
        return 1
    package = json.loads((RAW / "wfp" / "hdx_package.json").read_text(encoding="utf-8"))
    if wfp.published_through(package) < month:
        # Without this check a late price file would trigger the ARIMA fallback.
        print(f"WFP has not published {month} yet; run `nowcaster fetch wfp` later")
        return 1
    table = features.build(wfp.national_index(wfp.load(WFP_FILE))[0])
    forecasts = pd.read_csv(PROCESSED / "backtest.csv", index_col=0)
    forecasts.index = pd.PeriodIndex(forecasts.index, freq="M")
    abs_errors = forecasts.drop(columns="actual").sub(forecasts["actual"], axis=0).abs()
    food = target.division_2024(cpi2024, target.FOOD_2024)
    try:
        prediction = live.make(series, table, food, abs_errors)
    except ValueError as error:
        print(error)
        return 1
    prediction = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **prediction,
        "cpi_through": str(series.index[-1]),
        **_vintage(),
    }
    PREDICTIONS.mkdir(exist_ok=True)
    path.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    return 0


def _score(args: argparse.Namespace) -> int:
    series = target.food_mom(mospi.load_2012(RAW / "mospi"), mospi.load_2024(RAW / "mospi"))
    table = live.record(PREDICTIONS, series)
    table.to_csv(PREDICTIONS / "record.csv", index=False, lineterminator="\n")
    print(table.to_string(index=False))
    return 0


def _daily(args: argparse.Namespace) -> int:
    """One unattended run; safe to repeat. Commits and pushes are left to the caller."""
    polite, legacy = PoliteClient(), PoliteClient(legacy_tls=True)
    no_args = argparse.Namespace()
    failed = False
    try:
        try:
            print(f"dca: saved {dca.fetch(DCA_PAGES, polite)}")
            print(f"dca: {DCA_TABLE} has {dca.write_table(DCA_PAGES, DCA_TABLE)} rows")
        except (httpx.HTTPError, ValueError) as error:
            # A lost price day must not block a prediction; the run still ends as failed.
            print(f"dca: FAILED: {error}")
            failed = True

        def released(month: pd.Period) -> bool:
            return mospi.released(legacy, month)

        cpi_fetched = False
        month = daily.target_month(PREDICTIONS, released)
        wfp_through = wfp.published_through(polite.get(wfp.PACKAGE_URL).json())
        if not released(month - 1):
            print(f"predict: waiting for CPI for {month - 1}, which {month} is built on")
        elif wfp_through < month:
            print(f"predict: waiting for WFP prices for {month} (published through "
                  f"{wfp_through})")
        else:
            print(f"predict: CPI for {month} is not out and WFP has it; predicting")
            wfp.fetch(RAW / "wfp", polite)
            mospi.fetch(RAW / "mospi", legacy, date.today().year)
            cpi_fetched = True
            if _backtest(no_args) or _predict(no_args):
                return 1
            with (PREDICTIONS / "publication.csv").open("a", encoding="utf-8") as file:
                file.write(f"{month},{date.today()},true,committed and pushed by the daily "
                           "run before the release; the GitHub push time is the proof\n")

        to_score = [month for month in daily.unscored(PREDICTIONS) if released(month)]
        if not to_score:
            print("score: nothing newly released")
        else:
            print(f"score: CPI out for {', '.join(map(str, to_score))}")
            if not cpi_fetched:
                mospi.fetch(RAW / "mospi", legacy, date.today().year)
            failed |= bool(_score(no_args))
        return 1 if failed else 0
    finally:
        polite.close()
        legacy.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nowcaster", description=__doc__)
    parser.add_argument("--version", action="version", version=f"nowcaster {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download a source into data/raw/")
    fetch.add_argument("source", choices=["mospi", "dca", "wfp"])
    fetch.set_defaults(func=_fetch)
    sub.add_parser("check", help=_check.__doc__).set_defaults(func=_check)
    run = sub.add_parser("backtest", help="backtest baselines and price models")
    run.set_defaults(func=_backtest)
    predict = sub.add_parser("predict", help="forecast the next unreleased month")
    predict.set_defaults(func=_predict)
    sub.add_parser("score", help="score predictions against released CPI").set_defaults(
        func=_score
    )
    sub.add_parser("daily", help=_daily.__doc__).set_defaults(func=_daily)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
