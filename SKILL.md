---
name: basis-futures-strategy
description: Research/backtest pipeline for CSI stock-index-futures basis (IH/IF/IC/IM vs. 上证50/沪深300/中证500/中证1000) — dividend adjustment, factor construction, walk-forward regression, position-series backtesting, hedge-cost-ladder comparison, and scenario-based basis forecasting. Use when the user wants to run/extend/debug this basis backtest, add a data source, tune N/quantile/HAC parameters, reconcile raw-vs-dividend-adjusted results, or understand why a fixed/dynamic signal direction under/over-performs. Chinese keywords: 股指期货基差、贴水、升水、分红调整、walk-forward、对冲成本、IH/IF/IC/IM
---

# Basis Futures Strategy

A four-layer research pipeline for stock-index-futures basis:

```
data(数据层) → dividend(分红/结构层) → factors+regression(统计层)
    → strategy(策略/回测层) → forecast(预测层)
```

> **Positioning**: this is a research/backtest tool for analyzing basis structure, statistical
> regularities, and scenario forecasts. It is **not** an execution-ready trading system —
> every "signal"/"forecast"/"backtest return" is based on historical data and simplified
> assumptions (no slippage/impact/margin constraints by default). Independent validation and
> risk review are required before any live use.

## When to use this skill

Invoke it when the user wants to:
- Run the ten-year theoretical backtest (`python -m basis_model.run`) across all four indices (000016/000300/000905/000852).
- Add/switch a data source (akshare, local files, Wind) — see `basis_model/data.py`'s `DataAdapter` interface and `get_adapter()` factory.
- Reconcile raw-basis vs. dividend-adjusted results (`run_dividend_variants` → `compare_variants`).
- Compare fixed a-priori signal directions (deep-discount long / deep-premium long / etc.) against buy-and-hold and flat via `compare_signal_directions.py`.
- Run or debug the walk-forward dynamic-direction selection (`walkforward_direction.py`) — 3y train / 3mo test windows with purge/embargo and a pre-registered HAC-t / Sharpe(NW) gate.
- Investigate the two backtest engines in `strategy.py` and make sure the right one is used (see "Known limitations" below).
- Tune parameters in `basis_model/config.py` (N, quantile thresholds, cost, rolling windows).

## Pipeline (in execution order)

1. **`data.py`** — `DataAdapter` abstraction (`load_futures` / `load_index` / `load_dividends`) with four registered backends: `mock` (offline, tests only), `cffex_akshare` (CFFEX official monthly ZIP + akshare index, the raw-basis default), `local` (user-supplied CSV/XLSX/Parquet, see `input/local_data/README.md`), `akshare`, and a `wind` stub (`NotImplementedError` placeholder with TODOs for users with a Wind terminal license). All adapters produce an identical panel schema (`date, contract, index_code, month_tag, F, S, days_to_expiry`); switching sources only touches `config.DATA_SOURCE_PRIMARY`.
2. **`dividend.py`** — converts index dividend yield × constituent weight × index level into basis points-contribution, only accumulating events where `valuation_date < ex_date <= contract_expiry`. Supports an `assume_zero_dividend` toggle (raw basis vs. dividend-adjusted).
3. **`factors.py`** — builds the factor panel (annualized basis, month-dummy term structure, `f3_term_structure`), and enforces no-look-ahead via `assert_no_lookahead()` plus NaN-tailing the training-only columns.
4. **`regression.py`** — walk-forward OLS with Newey-West (HAC) t-stats, lag = N.
5. **`strategy.py`** — **two backtest engines, don't confuse them**: `backtest_position_series()` (primary — position-series method, non-overlapping daily settlement, HAC-adjusted Sharpe) vs. `backtest_naive_overlapping()` (illustrative only — overlapping N-day windows, double-counts returns, historically produced distorted numbers like +3892% annualized; carries a `note` field warning against comparison use). Also includes `hedge_cost_ladder()` (current-month/quarter-month/min-discount/regression-roll comparison).
6. **`forecast.py`** — returns a dict with `from_regression` (hard, statistical extrapolation, invariant to scenario assumptions — the function signature has no `future_return`/`y_convergence_rate` parameters, so leakage is a `TypeError` by construction) and `from_qualitative` (soft, behavioral overlay + scenario assumption).
7. **`report.py`** — optional docx report generation (`python-docx`); skipped with a printed notice if the dependency is missing.
8. **`run.py`** — orchestrates all four layers end-to-end for all four indices; per-index failures are caught/logged/skipped, never abort the whole run.

## Running it

```bash
pip install -r requirements.txt
python -m basis_model.run                       # ten-year raw-basis backtest, all 4 indices
python -m basis_model.run_dividend_variants      # requires input/local_dividend/ three tables
python -m basis_model.compare_variants
python -m basis_model.compare_signal_directions  # fixed a-priori directions vs. buy-and-hold/flat
python -m basis_model.walkforward_direction       # walk-forward dynamic direction selection
pytest tests/ -v
```

First run fetches CFFEX monthly ZIPs + akshare index data and caches them under `.basis_cache/` (gitignored, parquet with CSV fallback). Delete the cache directory to force a full refresh, or pass `refresh=True` to individual adapter calls.

## Key parameters (`basis_model/config.py`)

| Parameter | Meaning |
|---|---|
| `DATA_SOURCE_PRIMARY` / `DATA_SOURCE_FALLBACK` | Data source switch points (`cffex_akshare`, `akshare`, `local`, `wind`, `mock`) |
| `ASSUME_ZERO_DIVIDEND_DEFAULT` | `True` = raw basis (default); `False` = pulls per-year dividend events |
| Primary holding period `N` | 8 trading days (信达口径); robustness scan over `[5, 10, 20]` |
| `COST` | Zero by default — "theoretical" backtest, not net of slippage/fees |
| Signal-direction quantiles | 10%/90% over a trailing ~3y window, `shift(1)`'d to avoid same-day leakage |
| Walk-forward gates | HAC t-stat ≥ 2.13 (Bonferroni-corrected), Sharpe(NW) ≥ 0.5, else falls back to flat |

## Known limitations / do not use for

- No slippage, fees, margin constraints, position limits, or multi-contract concurrency management in either backtest engine — not a live-trading evaluation.
- `MockAdapter` produces zero-mean reproducible noise only, for tests/demos — has no seasonal dividend effect or discount prior, cannot be used to judge strategy validity.
- `assign_month_tags()` approximates CFFEX's "current/next month + two quarter months" listing rule by sorting live contracts by expiry — may diverge from official labels around new-listing/rollover edge cases.
- `dividend.py`'s point-conversion formula is a research approximation (yield × weight × index level), not aligned with the index provider's actual divisor-adjustment mechanism.
- `hedge_cost_ladder()`'s "regression roll" rule calls `regression.predict_convergence_rate()` in a per-row Python loop (not vectorized) — acceptable at ten-year × four-contract scale, don't reuse as-is at higher frequency/larger candidate sets.
- Reserved-but-unimplemented hooks (raise `NotImplementedError`): `regression.ou_mean_reversion_params()`, `regression.kalman_dynamic_coefficients()`, `regression.walk_forward_oos_validation()`; `forecast.behavioral_adjust()` is a minimal linear-heuristic stub pending an explicit fund-flow model.
