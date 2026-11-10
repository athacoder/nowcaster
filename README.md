# nowcaster

Predict India's monthly food inflation before the official number comes out.

The Ministry of Statistics (MoSPI) publishes each month's Consumer Price Index around the
12th of the next month. Retail food prices are published every day. This project uses
the daily prices to estimate the month's change in the CPI Food and Beverages index
before the release, and scores every method against simple baselines.

**Status:** data loaders and baselines are done. Machine-learning models and live
predictions are not built yet.

## Results so far

Target: month-on-month change (percent) in the all-India Food and Beverages index.
Backtest: 129 one-month-ahead forecasts, January 2016 to September 2026. Each forecast
uses only CPI released before the target month's own release.

| Method | MAE (pp) | RMSE (pp) | MAE vs seasonal-naive |
|---|---|---|---|
| Seasonal-naive (same month last year) | 1.064 | 1.380 | 1.00 |
| Seasonal mean (same month, all earlier years) | 0.739 | 0.975 | 0.69 |
| ARIMA, airline model | 0.675 | 0.942 | 0.63 |

Both better baselines beat seasonal-naive clearly (Diebold-Mariano p < 0.00001). ARIMA
against the seasonal mean is borderline (p = 0.048). ARIMA is the bar a price-based model
has to clear; beating seasonal-naive alone would prove little.

## Run it

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # bin/ instead of Scripts/ off Windows
nowcaster fetch mospi     # CPI, about 5 minutes at one request every 2 seconds
nowcaster fetch wfp       # monthly retail prices, 26 MB
nowcaster fetch dca       # today's all-India average prices
nowcaster check           # validate the commodity-to-CPI map
nowcaster backtest        # the table above
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
- The price sources cover staples, pulses, oils and three vegetables. They do not cover
  meat, fish, fruit other than banana, or most vegetables.

## Licence

Code: MIT. Data stays with its publishers; see the data sources file.
