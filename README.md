# nowcaster

Predict India's monthly food inflation before the official number comes out.

The Ministry of Statistics (MoSPI) publishes each month's Consumer Price Index around the
12th of the next month. Retail food prices are published every day. This project uses
the daily prices to estimate the month's change in the CPI Food and Beverages index
before the release, and scores every method against simple baselines.

**Status:** baselines and price-based models are backtested. Intervals and live
predictions are not built yet.

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
| Seasonal mean + ridge on 28 price features | 0.692 | 0.908 | 1.02 | 0.66 | 51 |
| ARIMA + ridge on 28 price features | 0.634 | 0.892 | 0.94 | 0.17 | 69 |
| **ARIMA + tomato, onion, potato prices** | **0.495** | **0.663** | **0.73** | **0.0015** | **75** |

p-values are from a Diebold-Mariano test on absolute error.

What this says:

- Three vegetable prices cut ARIMA's error by 27%. Against seasonal-naive the cut is 53%.
- Adding 25 more price series through a ridge model did not help. With about 130 months
  of data, the small model wins.
- **The gain does not show on the new CPI base yet.** Over the nine months of 2026 the
  vegetable model's MAE is 0.588 against ARIMA's 0.371 (p = 0.28). Nine months cannot
  settle it either way, but the headline number is mostly a result about the old basket.
- In 10 of the 129 months the price file has no vegetable prices. The model then treats
  prices as unchanged and does worse than ARIMA (0.591 against 0.314). On the other 119
  months it scores 0.486 against 0.706.

The model list and the feature list were fixed before any model was scored. Gradient
boosting was planned only if the ridge models showed a gain; they did not, so it was not
run.

## Run it

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # bin/ instead of Scripts/ off Windows
nowcaster fetch mospi     # CPI, about 5 minutes at one request every 2 seconds
nowcaster fetch wfp       # monthly retail prices, 26 MB
nowcaster fetch dca       # today's all-India average prices
nowcaster check           # validate the commodity-to-CPI map
nowcaster backtest        # the table above; about 4 minutes
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
  Dropping that month changes little (0.487 against 0.658).
- The price sources cover staples, pulses, oils and three vegetables. They do not cover
  meat, fish, fruit other than banana, or most vegetables.

## Licence

Code: MIT. Data stays with its publishers; see the data sources file.
