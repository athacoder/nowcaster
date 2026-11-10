# Data sources

Checked on 2026-11-10. Every request used the User-Agent
`food-nowcaster/0.1 (student research project; low-rate)` and at most one request every
two seconds per host. Nothing here is fetched past a CAPTCHA, a login or a robots.txt
rule.

## In use

### CPI: MoSPI open API

- **URL:** `https://api.mospi.gov.in/api/cpi/` (`getCPIIndex` for base 2012, `getCPIData`
  for base 2024). No key needed. The Swagger file is in MoSPI's own repository,
  `nso-india/esankhyiki-mcp`.
- **Format:** JSON, at most 100 rows per page.
- **Coverage:**
  - Base 2024, current series: January 2025 to September 2026, 12 divisions down to 358
    items, by state and sector. Food and beverages has 148 items.
  - Base 2024, back series: January 2013 to December 2024, **General index only**.
  - Base 2012: January 2011 to December 2025; groups and subgroups (Food and Beverages
    has 12 subgroups) plus the Consumer Food Price Index.
- **Update lag:** the month's CPI comes out on the 12th of the next month, or the next
  working day (2026 dates on MoSPI's release calendar: 12 Jan, 12 Feb, 12 Mar, 13 Apr,
  12 May, 12 Jun, 13 Jul, 12 Aug, 14 Sep, 12 Oct).
- **Terms:** robots.txt on `esankhyiki.mospi.gov.in` allows all agents; the API is
  published for programmatic use. No licence text was found on the API itself.
- **Automated download:** allowed.
- **Quirks found:**
  - No official food series spans the base change. Joining the index levels by the ratio
    of 2025 averages (MoSPI's method for General; it reproduces their factor 0.5267 and
    their back series to within 0.06 index points) puts a false drop at January 2025 for
    food: -3.65% against the published -2.46%. The target is therefore built from
    month-on-month changes, each taken inside one series.
  - The server needs TLS legacy renegotiation, which OpenSSL 3 refuses by default.
  - `getCPIIndex` hides rows at page boundaries: a query of 180 rows returned 178 distinct
    rows on three passes in a row. Requests are cut to half-years so each fits one page,
    and every download checks that the distinct row count equals the API's own total.

### Retail prices, history: WFP food prices for India (HDX)

- **URL:** `https://data.humdata.org/dataset/wfp-food-prices-for-india`
- **Format:** one CSV, 25.9 MB, 216,151 rows.
- **Coverage:** monthly averages, January 1994 to October 2026. The source is the
  Department of Consumer Affairs price monitoring data. 21 commodities have history from
  2012-2015; 18 more start in January 2026.
- **Update lag:** the file modified on 5 November 2026 already held October 2026.
- **Licence:** CC BY 3.0 IGO. Attribution: World Food Programme.
- **Automated download:** allowed (HDX offers a CKAN API).
- **Quirks found:** the markets change. 18 cities in 1994, about 160 centres in 2022, and from
  August 2023 only five zone averages. No market spans July-August 2023, so that one
  month-on-month change is taken from cross-market means and is flagged.

### Retail and wholesale prices, daily: Department of Consumer Affairs

- **URL:** `https://fcainfoweb.nic.in/Reports/DB/Dailyprices.aspx`, linked from the
  department's price monitoring page.
- **Format:** HTML table.
- **Coverage:** all-India average for 42 retail and 24 wholesale commodities: today plus
  1-5 days, 1 week, 1 month and 1 year back. No history beyond that.
- **Update lag:** same day. Weekend and holiday figures are provisional until the next
  working day (the site says so).
- **Terms:** the site's disclaimer covers accuracy and liability only; no licence and no
  rule on reuse or automated access. The host serves no robots.txt (404).
- **Automated download:** not forbidden; one request a day is enough.

## Blocked

### Department of Consumer Affairs, full daily history

`https://fcainfoweb.nic.in/Reports/Report_Menu_Web.aspx` asks for a CAPTCHA before any
report. Not fetched. Fallbacks: the WFP file above (monthly), the daily page going
forward, or a data.gov.in API key (below).

### Agmarknet mandi prices

`https://agmarknet.gov.in/daily-price-and-arrival-report` (Agmarknet 2.0) asks for a
CAPTCHA before any report. Not fetched. Its robots.txt allows the public pages and its
copyright policy allows reuse with the source acknowledged, but the CAPTCHA rules out
automated download.

Fallbacks, in order:

1. **data.gov.in API.** Agmarknet's daily mandi prices are published there. It needs a
   free API key tied to a personal account; without one the API answers
   `Authorization field missing`. Not tested further. `www.data.gov.in` disallows all
   crawlers in robots.txt, so the catalogue pages were not read.
2. **CEDA Agri Market Data** (`agmarknet.ceda.ashoka.edu.in`). Monthly and daily mandi
   prices by commodity with a download button; non-commercial use with citation. It has
   no documented API, so only manual downloads.

## Not yet in use

### News: GDELT DOC 2.0 API

- **URL:** `https://api.gdeltproject.org/api/v2/doc/doc`
- **Format:** JSON list of article URL, title, date, domain, language. Titles only.
- **Coverage:** a query starting in February 2016 was refused as too early; one for
  September 2019 worked.
- **Terms:** not read yet. The API's own error message asks for at most one request
  every five seconds.
- **Automated download:** offered as an API, but of five valid requests spaced 8 seconds
  to several minutes apart, four returned HTTP 429. A collector will have to run much
  slower than the stated limit. No sample is saved yet.
