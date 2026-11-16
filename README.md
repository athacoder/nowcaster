# nowcaster

Predict India's monthly food inflation before the official number comes out.

The Ministry of Statistics (MoSPI) publishes each month's Consumer Price Index around the
12th of the next month. Retail food prices are published every day. This project uses
the daily prices to estimate the month's change in the CPI Food and Beverages index
before the release, and scores every method against simple baselines.

**Status:** backtested, with calibrated intervals. The first live prediction (October
2026) is in [predictions/](predictions/).

## Results so far

Target: month-on-month change (percent) in the all-India Food and Beverages index.
Backtest: 129 one-month-ahead forecasts, January 2016 to September 2026. Each forecast
uses CPI released before the target month's own release, and retail prices up to the end
of the target month.

| Method | MAE (pp) | RMSE (pp) | MAE vs ARIMA | p vs ARIMA | Months won of 129 |
|---|---|---|---|---|---|
| Seasonal-naive (same month last year) | 1.064 | 1.380 | 1.58 | <0.001 | 37 |
| Seasonal mean (same month, earlier years) | 0.739 | 0.975 | 1.10 | 0.048 | 45 |
| ARIMA, airline model | 0.675 | 0.942 | 1.00 | - | - |
| Seasonal mean + ridge on 28 price features | 0.702 | 0.919 | 1.04 | 0.46 | 50 |
| ARIMA + ridge on 28 price features | 0.646 | 0.905 | 0.96 | 0.30 | 67 |
| **ARIMA + tomato, onion, potato prices** | **0.519** | **0.718** | **0.77** | **0.0054** | **74** |

p-values are from a Diebold-Mariano test on absolute error.

What this says:

- Three vegetable prices cut ARIMA's error by 23%, and seasonal-naive's by 51%.
- Adding 25 more price series through a ridge model did not help. With about 130 months
  of data, the small model wins.
- **The gain does not show on the new CPI base yet.** Over the nine months of 2026 the
  vegetable model's MAE is 0.620 against ARIMA's 0.371 (p = 0.15). Five of those nine
  months have no real price data (see Limits); there the model sees "no change" and
  undershoots. `nowcaster predict` refuses to run in such a month.

The model list and the feature list were fixed before any model was scored. Gradient
boosting was planned only if the ridge models showed a gain; they did not, so it was not
run.

### Intervals

Each interval is the forecast plus or minus a quantile of the model's own absolute errors
in earlier months (split conformal, no normal assumption). Checked on the 93 backtest
months after the first 36:

| Model | 80% interval: covered | width (pp) | 95% interval: covered | width (pp) |
|---|---|---|---|---|
| ARIMA | 76.3% | 1.90 | 94.6% | 5.04 |
| ARIMA + vegetable prices | 76.3% | 1.59 | 95.7% | 2.94 |

Coverage is within sampling noise of the target (about 4 points either way at 93
months), and the vegetable model's 95% interval is 42% narrower than ARIMA's.

## Live predictions

Each prediction is written to `predictions/YYYY-MM.json` before MoSPI releases that
month's CPI, with the code commit and a hash of the price file it used. A prediction file
is never overwritten. `nowcaster score` compares them with the released numbers in
`predictions/record.csv`.

A prediction counts only if it was pushed here before the release. Publication dates are
in `predictions/publication.csv`.

| Month | Made (UTC) | Forecast, m-o-m | 80% interval | Implied y-o-y | Actual m-o-m | Actual y-o-y | Error | Verified before release |
|---|---|---|---|---|---|---|---|---|
| October 2026 | 2026-11-11 04:57 (local clock) | +0.84% | -0.01% to +1.68% | 8.44% | +0.90% | 8.51% | 0.06 pp | **No** |

October was made before the release but first pushed on 16 November, after it. Its
timestamp comes from a local commit and cannot be checked, so it is not part of the
verified record. The first prediction that can be verified will be November 2026.

## Run it

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # bin/ instead of Scripts/ off Windows
nowcaster fetch mospi     # CPI, about 5 minutes at one request every 2 seconds
nowcaster fetch wfp       # monthly retail prices, 26 MB
nowcaster fetch dca       # today's all-India average prices
nowcaster check           # validate the commodity-to-CPI map
nowcaster backtest        # the tables above; about 4 minutes
nowcaster predict         # next unreleased month, into predictions/
nowcaster score           # after the release
```

Checks: `ruff check src tests`, `mypy`, `pytest`.

## Data

[docs/data-sources.md](docs/data-sources.md) lists every source with its coverage,
terms, and what could not be fetched. In short: CPI comes from MoSPI's open API; price
history from the World Food Programme's India file (CC BY 3.0 IGO), which republishes
Department of Consumer Affairs prices as monthly averages; daily prices from the
department's public page. The department's full daily history and Agmarknet mandi prices
sit behind CAPTCHAs and are not used.

[data/reference/commodity_map.csv](data/reference/commodity_map.csv) maps the 42
commodities on the daily price page to CPI items: 41 map to 38 of the 148 food items.

## Limits

- The CPI base changed from 2012 to 2024 in February 2026, with a new basket and weights.
  No official food series spans the change, so the target takes each month's change from
  the series that was official at the time. Over 2025, when both exist, their monthly
  changes differ by 0.16 points on average.
- The backtest uses today's CPI figures, not the figures as first published.
- It assumes a month's average retail prices are known at month end. The underlying
  prices are published daily; the monthly file's own publication date was not checked.
- The price file switches from about 160 centres to five zone averages in August 2023.
- WFP fills some months by copying the month before: every commodity, every zone, in
  2024-11, 2024-12, 2026-04, 2026-07 and 2026-08. These are treated as missing. Before
  this was caught the vegetable model scored 0.495; the corrected figure is 0.519.
- The price sources cover staples, pulses, oils and three vegetables. They do not cover
  meat, fish, fruit other than banana, or most vegetables.

## Licence

Code: MIT. Data stays with its publishers; see the data sources file.
