"""用因果阈值比较三个固定方向信号与两个基准，不做历史择优。"""

from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from basis_model import config, factors, strategy

logger = logging.getLogger("basis_model.compare_signal_directions")

STRATEGIES = {
    "fixed_momentum_long": "momentum_long",
    "fixed_premium_long": "premium_long",
    "fixed_two_sided": "two_sided",
}
METRIC_KEYS = (
    "annualized_return", "annualized_vol", "sharpe_nw", "max_drawdown",
    "win_rate", "n_trades", "signal_tstat_hac",
)


def evaluate_fixed_directions(
    factor_panel: pd.DataFrame,
    index_code: str,
    variant: str,
    cfg: config.BacktestConfig = config.DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """在同一日期集合上评估三信号和 buy-and-hold/flat。"""
    front = (
        factor_panel[factor_panel["month_tag"] == "当月"]
        .sort_values("date").drop_duplicates("date").copy()
    )
    continuous = pd.Series("CONTINUOUS_FRONT", index=front.index)
    ret_1d = factors.basis_daily_return(
        front["adj_basis"], front["S"], continuous, front["date"]
    )
    signals = {
        name: strategy.fixed_direction_signal(
            front["f1_ann_basis"], continuous, front["date"],
            window=cfg.signal_direction_window,
            pct=cfg.signal_direction_tail_pct,
            min_periods=cfg.signal_direction_min_periods,
            rule=rule,
        )
        for name, rule in STRATEGIES.items()
    }
    signals["buy_and_hold"] = pd.Series(1, index=front.index, dtype=int)
    signals["flat"] = pd.Series(0, index=front.index, dtype=int)

    rows = []
    for name, signal in signals.items():
        result = strategy.backtest_position_series(
            dates=front["date"], signal=signal, ret_1d=ret_1d,
            N=cfg.n_horizon, cost=cfg.cost,
        )
        row = {
            "basis_variant": variant,
            "index_code": index_code,
            "strategy": name,
            "n_dates": len(front),
            "signal_nonzero_days": int(signal.ne(0).sum()),
        }
        row.update({key: result[key] for key in METRIC_KEYS})
        rows.append(row)
    output = pd.DataFrame(rows)
    buy_return = float(output.loc[output["strategy"] == "buy_and_hold", "annualized_return"].iloc[0])
    output["beats_flat"] = output["annualized_return"] > 0.0
    output["beats_buy_and_hold"] = output["annualized_return"] > buy_return
    output["causal_threshold"] = True
    output["direction_selected_from_returns"] = False
    return output, signals


def run_comparison(output_root: str | Path) -> pd.DataFrame:
    root = Path(output_root)
    variants = (("raw_basis", "raw_basis"), ("announced_dividend", "dividend_adjusted"))
    all_results = []
    raw_keys: dict[str, pd.DataFrame] = {}
    for variant, dirname in variants:
        variant_dir = root / dirname
        result_dir = variant_dir / "fixed_signals"
        summary_path = variant_dir / "summary.csv"
        if variant == "announced_dividend":
            manifest_path = root / dirname / "run_manifest.csv"
            if not manifest_path.exists():
                logger.warning("announced_dividend 缺少资格清单，跳过该口径。")
                continue
            manifest = pd.read_csv(manifest_path)
            eligible = manifest.get(
                "formal_backtest_eligible", pd.Series([False])
            ).fillna(False).astype(bool).all()
            if not eligible:
                logger.warning("announced_dividend 未通过 point-in-time 资格验收，跳过该口径。")
                continue
        if not summary_path.exists():
            logger.warning("%s 尚无结果，跳过该口径。", variant)
            continue
        summary = pd.read_csv(summary_path, dtype={"index_code": str})
        variant_results = []
        for item in summary.itertuples(index=False):
            code = str(item.index_code).zfill(6)
            path = root / dirname / "indices" / f"{code}_{item.name}" / "factor_panel.csv"
            panel = pd.read_csv(path, parse_dates=["date"], dtype={"index_code": str})
            keys = panel[["date", "contract"]].reset_index(drop=True)
            if variant == "raw_basis":
                raw_keys[code] = keys
            elif code not in raw_keys or not keys.equals(raw_keys[code]):
                raise ValueError(f"{code} announced 与 raw 面板日期/合约不一致，禁止比较")
            result, _ = evaluate_fixed_directions(panel, code, variant)
            result.insert(2, "name", item.name)
            result_dir.mkdir(parents=True, exist_ok=True)
            result.to_csv(result_dir / f"{code}_{item.name}.csv", index=False)
            variant_results.append(result)
            all_results.append(result)
        if variant_results:
            pd.concat(variant_results, ignore_index=True).to_csv(
                result_dir / "summary.csv", index=False
            )
    if not all_results:
        raise FileNotFoundError("raw/announced 均无可用因子面板")
    combined = pd.concat(all_results, ignore_index=True)
    comparison_dir = root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(comparison_dir / "fixed_signals_all_variants.csv", index=False)
    return combined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(__file__).resolve().parent.parent / "output"
    result = run_comparison(root)
    display = result[[
        "basis_variant", "index_code", "name", "strategy", "annualized_return",
        "sharpe_nw", "max_drawdown", "beats_flat", "beats_buy_and_hold",
    ]]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
