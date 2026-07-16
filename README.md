# Basis Futures Strategy

A research/backtest framework for CSI stock-index-futures basis — IH/IF/IC/IM futures against
上证50 (000016) / 沪深300 (000300) / 中证500 (000905) / 中证1000 (000852). Four layers share one
pipeline: dividend adjustment → factor construction → walk-forward regression → position-series
backtest → scenario-based forecast. See [SKILL.md](SKILL.md) for the full pipeline description,
parameters, and known limitations.

> **Positioning**: this is a research/backtest tool, not an execution-ready trading system. All
> signals, forecasts, and backtest returns come from historical data and simplified assumptions
> (no slippage/fees/margin constraints by default) — independent validation and risk review are
> required before any live use. See the Disclaimer below.

## Use as a Claude Code Skill

Clone this repo into your Claude Code skills directory (or a project's `.claude/skills/`) so Claude can pick it up:

```bash
git clone <this-repo-url> ~/.claude/skills/basis-futures-strategy
```

Claude will use `SKILL.md` to decide when to invoke it — e.g. when you ask it to run the basis backtest, switch data sources, tune N/quantile parameters, or explain why a fixed vs. dynamic signal direction under/over-performs.

## Run it standalone

```bash
pip install -r requirements.txt
python -m basis_model.run
```

Futures prices are read from CFFEX's official monthly ZIPs; spot index levels come from
[akshare](https://github.com/akfamily/akshare) (free, public data). Everything is cached locally
under `.basis_cache/` (gitignored) — no data files are bundled in this repo. Outputs land under
`output/` (gitignored; see `output/README.md` for the expected structure).

To use your own market data instead of akshare, see `input/local_data/README.md` and set
`DATA_SOURCE_PRIMARY = "local"` in `basis_model/config.py`. To run the dividend-adjusted variant
with point-in-time dividend/weight/price tables, see `input/local_dividend/README.md`.

## Three disciplines (enforced in code, not just docs)

1. **Strict ordering**: strip dividends → build factors → regress → map to strategy → forecast. `run.py` calls the four layers in exactly this order.
2. **No look-ahead**: `f2_future_return` / `y_convergence_rate` may only enter `regression.py`'s training matrix — `forecast.py`'s function signatures don't even accept these parameters (a leak is a `TypeError`, not a bug you'd discover in backtest numbers). `factors.assert_no_lookahead()` adds a defensive assertion; `tests/test_regression_leakage.py` covers this.
3. **Hard/soft separation**: `forecast.forecast_basis()` always returns both `from_regression` (hard, pure statistical extrapolation, invariant to scenario assumptions) and `from_qualitative` (soft, behavioral overlay + scenario assumption).
4. **Data-source isolation**: only `data.py` may `import akshare` — every other module (factors/regression/strategy/forecast/run) depends solely on the `DataAdapter` abstraction and the standard panel schema. `tests/test_data_source_isolation.py` statically scans imports to enforce this. Switching data sources (e.g. akshare → Wind) only touches `data.get_adapter(name)`'s `name` argument.

## Layout

```
basis_model/
  config.py                    # every tunable parameter (indices, N, dates, cost, data source)
  data.py                      # DataAdapter interface + mock/cffex_akshare/akshare/local/wind backends
  dividend.py                  # dividend-adjustment core identity
  factors.py                   # factor panel construction + no-look-ahead assertions
  regression.py                # walk-forward OLS with Newey-West (HAC) t-stats
  strategy.py                  # backtest_position_series() [primary] + backtest_naive_overlapping() [illustrative only] + hedge_cost_ladder()
  forecast.py                  # hard (regression) + soft (qualitative) basis forecast
  report.py                    # optional docx report (python-docx)
  run.py                       # end-to-end entry point, all 4 indices
  run_dividend_variants.py     # dividend-adjusted variant using local point-in-time tables
  compare_variants.py          # raw vs. dividend-adjusted reconciliation
  compare_signal_directions.py # fixed a-priori signal directions vs. buy-and-hold/flat
  walkforward_direction.py     # walk-forward dynamic direction selection (purge/embargo/HAC gate)
  validate_local_data.py       # validates input/local_data/ before switching to DATA_SOURCE_PRIMARY="local"
tests/                          # pytest suite — see SKILL.md for what each file covers
input/
  local_data/README.md          # format spec for bring-your-own market data
  local_dividend/README.md      # format spec for point-in-time dividend/weight/price tables
output/README.md                # expected output structure (generated files are gitignored)
```

## Data source switching (akshare → Wind, etc.)

All sources implement the same `DataAdapter` interface (`load_futures` / `load_index` /
`load_dividends`) and produce an identical panel schema. Register a new source in
`data._ADAPTER_REGISTRY` and point `config.DATA_SOURCE_PRIMARY` at it — no changes needed in
factors/regression/strategy/forecast/run. A `wind` adapter stub already exists in `data.py` with
detailed TODOs for anyone with a Wind terminal license; `run.resolve_primary_adapter()` detects
the stub and falls back to `DATA_SOURCE_FALLBACK` automatically.

## Tests

```bash
pytest tests/ -v
```

Covers expiry-date calculation, CFFEX ZIP parsing, the dividend-adjustment identity, look-ahead
prevention, the two backtest engines' divergence (and why), data-source import isolation, and a
regression test for a real bug (`MockAdapter` mislabeling expired contracts as "current month").
Full list in [SKILL.md](SKILL.md).

## Methodology reference

The core idea — decomposing the basis into an annualized-carry term plus month-of-expiry
dummies, walk-forward re-estimating the loading, and mapping the fitted coefficient into a
position via pre-registered thresholds — is the researcher's own reimplementation of standard
sell-side basis-trading methodology; no proprietary report text, tables, or vendor data are
reproduced here.

## Disclaimer

Research code for educational/personal use. Not investment advice; no warranty on data
correctness or backtest performance. See [SKILL.md](SKILL.md) "Known limitations" for specifics
on what this framework does *not* model (slippage, fees, margin, position limits, multi-contract
concurrency).
