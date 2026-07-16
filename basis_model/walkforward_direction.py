"""训练期动态选向的 purged + embargoed walk-forward 纯样本外回测。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from basis_model import config, factors, strategy

logger = logging.getLogger("basis_model.walkforward_direction")

CANDIDATE_RULES = {
    "fixed_momentum_long": "momentum_long",
    "fixed_premium_long": "premium_long",
    "fixed_two_sided": "two_sided",
}
REPORT_STRATEGIES = (*CANDIDATE_RULES, "walkforward_dynamic", "buy_and_hold", "flat")
METRIC_KEYS = (
    "annualized_return", "annualized_vol", "sharpe_nw", "max_drawdown",
    "win_rate", "n_trades", "signal_tstat_hac",
)


def _front_inputs(
    factor_panel: pd.DataFrame,
    cfg: config.BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series]:
    """构造一次连续当月序列、三个因果固定信号和单日基差收益。"""
    front = (
        factor_panel[factor_panel["month_tag"] == "当月"]
        .sort_values("date").drop_duplicates("date").reset_index(drop=True).copy()
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
        for name, rule in CANDIDATE_RULES.items()
    }
    signals["flat"] = pd.Series(0, index=front.index, dtype=int)
    return front, signals, ret_1d


def _position_from_signal(signal: pd.Series, n_horizon: int) -> pd.Series:
    """在一个独立窗口内把触发信号转换为最多持有 N 日的仓位。"""
    hold = max(n_horizon - 1, 0)
    position = signal.reset_index(drop=True).where(signal.reset_index(drop=True).ne(0), np.nan)
    return position.ffill(limit=hold).fillna(0.0).clip(-1.0, 1.0)


def select_direction_from_training(
    train_dates: pd.Series,
    train_returns: pd.Series,
    train_signals: dict[str, pd.Series],
    cfg: config.BacktestConfig = config.DEFAULT_CONFIG,
) -> tuple[str, str, pd.DataFrame]:
    """只接收 purge 后训练数据，按 Sharpe_NW 排序并应用预注册门槛。

    函数签名刻意不接收测试收益，保证测试表现无法参与方向选择。三个非 flat
    候选使用固定 N=8、10%/90% 因果阈值；HAC t 门槛 2.13 对应三个候选的
    一侧 5% Bonferroni 修正，另要求 Sharpe_NW >= 0.5。
    """
    rows = []
    for name in CANDIDATE_RULES:
        result = strategy.backtest_position_series(
            dates=train_dates.reset_index(drop=True),
            signal=train_signals[name].reset_index(drop=True),
            ret_1d=train_returns.reset_index(drop=True),
            N=cfg.n_horizon,
            cost=cfg.cost,
        )
        rows.append({"candidate": name, **{key: result[key] for key in METRIC_KEYS}})
    metrics = pd.DataFrame(rows)
    ranked = metrics.sort_values(
        ["sharpe_nw", "signal_tstat_hac", "candidate"],
        ascending=[False, False, True], na_position="last",
    )
    best = str(ranked.iloc[0]["candidate"])
    best_row = ranked.iloc[0]
    passed = bool(
        pd.notna(best_row["signal_tstat_hac"])
        and best_row["signal_tstat_hac"] >= cfg.walkforward_hac_tstat_min
        and best_row["sharpe_nw"] >= cfg.walkforward_sharpe_nw_min
    )
    selected = best if passed else "flat"
    metrics["passes_gate"] = (
        metrics["signal_tstat_hac"].ge(cfg.walkforward_hac_tstat_min)
        & metrics["sharpe_nw"].ge(cfg.walkforward_sharpe_nw_min)
    )
    return selected, best, metrics


def _window_trade_count(position: pd.Series) -> int:
    return int(position.diff().fillna(position).ne(0).sum())


def run_walkforward(
    factor_panel: pd.DataFrame,
    index_code: str,
    variant: str,
    cfg: config.BacktestConfig = config.DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行单指数动态选向，并在完全相同 OOS 日期上评估六种策略。"""
    front, signals, ret_1d = _front_inputs(factor_panel, cfg)
    dates = pd.to_datetime(front["date"]).reset_index(drop=True)
    if dates.empty:
        raise ValueError(f"{index_code} 无当月合约数据")

    origin = dates.iloc[0] + pd.DateOffset(years=cfg.walkforward_train_years)
    final_date = dates.iloc[-1]
    window_frames: list[pd.DataFrame] = []
    log_rows: list[dict] = []
    window_id = 0

    while origin <= final_date:
        train_start = origin - pd.DateOffset(years=cfg.walkforward_train_years)
        test_end = origin + pd.DateOffset(months=cfg.walkforward_test_months)
        nominal_train = np.flatnonzero((dates >= train_start) & (dates < origin))
        nominal_test = np.flatnonzero((dates >= origin) & (dates < test_end))
        if len(nominal_train) > cfg.n_horizon and len(nominal_test) > cfg.n_horizon:
            # 双向隔离：训练末尾 purge N 个交易日；测试开头 embargo N 个交易日。
            train_idx = nominal_train[:-cfg.n_horizon]
            oos_idx = nominal_test[cfg.n_horizon:]
            train_signals = {name: sig.iloc[train_idx] for name, sig in signals.items()}
            selected, best, train_metrics = select_direction_from_training(
                dates.iloc[train_idx], ret_1d.iloc[train_idx], train_signals, cfg
            )

            frame = pd.DataFrame({
                "date": dates.iloc[oos_idx].to_numpy(),
                "ret_1d": ret_1d.iloc[oos_idx].to_numpy(),
                "window_id": window_id,
            })
            for name in CANDIDATE_RULES:
                frame[name] = _position_from_signal(signals[name].iloc[oos_idx], cfg.n_horizon)
            frame["walkforward_dynamic"] = (
                _position_from_signal(signals[selected].iloc[oos_idx], cfg.n_horizon)
                if selected != "flat" else 0.0
            )
            frame["buy_and_hold"] = 1.0
            frame["flat"] = 0.0
            window_frames.append(frame)

            best_row = train_metrics.loc[train_metrics["candidate"].eq(best)].iloc[0]
            metric_map = train_metrics.set_index("candidate")
            log_row = {
                "window_id": window_id,
                "nominal_train_start": dates.iloc[nominal_train[0]],
                "nominal_train_end": dates.iloc[nominal_train[-1]],
                "train_start_after_windowing": dates.iloc[train_idx[0]],
                "train_end_after_purge": dates.iloc[train_idx[-1]],
                "purge_n_trading_days": cfg.n_horizon,
                "test_start": dates.iloc[nominal_test[0]],
                "embargo_end": dates.iloc[nominal_test[cfg.n_horizon - 1]],
                "oos_start": dates.iloc[oos_idx[0]],
                "oos_end": dates.iloc[oos_idx[-1]],
                "test_end_exclusive": test_end,
                "embargo_n_trading_days": cfg.n_horizon,
                "n_train_nominal": len(nominal_train),
                "n_train_after_purge": len(train_idx),
                "n_test_nominal": len(nominal_test),
                "n_test_after_embargo": len(oos_idx),
                "best_candidate": best,
                "selected_direction": selected,
                "selected_flat": selected == "flat",
                "best_train_sharpe_nw": best_row["sharpe_nw"],
                "best_train_tstat_hac": best_row["signal_tstat_hac"],
                "gate_tstat_min": cfg.walkforward_hac_tstat_min,
                "gate_sharpe_nw_min": cfg.walkforward_sharpe_nw_min,
                "multiple_testing_control": "Bonferroni(3), one-sided alpha=5%",
                "parameter_scan": False,
            }
            for name in CANDIDATE_RULES:
                log_row[f"train_sharpe_nw__{name}"] = metric_map.loc[name, "sharpe_nw"]
                log_row[f"train_tstat_hac__{name}"] = metric_map.loc[name, "signal_tstat_hac"]
            log_rows.append(log_row)
            window_id += 1
        origin += pd.DateOffset(months=cfg.walkforward_test_months)

    if not window_frames:
        raise ValueError(f"{index_code} 历史不足以形成 3年训练/3月测试窗口")
    oos = pd.concat(window_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if oos["date"].duplicated().any():
        raise AssertionError("walk-forward 测试窗口发生重叠")
    log = pd.DataFrame(log_rows)
    exploratory = bool(index_code == "000852" or len(log) < 8)

    result_rows = []
    equity_frames = []
    for name in REPORT_STRATEGIES:
        position = oos[name].astype(float)
        result = strategy.backtest_position_series(
            dates=oos["date"], signal=position, ret_1d=oos["ret_1d"],
            N=cfg.n_horizon, cost=cfg.cost, position_override=position,
        )
        # 每个测试窗口在 embargo 后从 flat 重启，交易数也按窗口分别计算。
        n_trades = sum(_window_trade_count(g[name]) for _, g in oos.groupby("window_id"))
        row = {
            "basis_variant": variant,
            "index_code": index_code,
            "strategy": name,
            "n_oos_dates": len(oos),
            "n_test_windows": len(log),
            "exploratory": exploratory,
            "confidence_label": "exploratory" if exploratory else "standard",
            "purge_days": cfg.n_horizon,
            "embargo_days": cfg.n_horizon,
            "parameter_scan": False,
            "multiple_testing_control": "Bonferroni(3)",
        }
        row.update({key: result[key] for key in METRIC_KEYS})
        row["n_trades"] = n_trades
        result_rows.append(row)
        equity_frames.append(result["equity_curve"].assign(strategy=name))

    counts = (
        log["selected_direction"].value_counts().reindex(
            [*CANDIDATE_RULES, "flat"], fill_value=0
        ).rename_axis("selected_direction").reset_index(name="n_windows")
    )
    counts["share"] = counts["n_windows"] / len(log)
    counts.insert(0, "index_code", index_code)
    counts.insert(0, "basis_variant", variant)
    equity = pd.concat(equity_frames, ignore_index=True)
    return pd.DataFrame(result_rows), log, counts, equity


def apply_ic_selection_to_im(
    im_panel: pd.DataFrame,
    ic_log: pd.DataFrame,
    variant: str,
    cfg: config.BacktestConfig = config.DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 IC 训练期选出的方向固定应用到 IM，IM 收益不参与选择。"""
    front, signals, ret_1d = _front_inputs(im_panel, cfg)
    dates = pd.to_datetime(front["date"]).reset_index(drop=True)
    frames = []
    transfer_log = []
    for row in ic_log.itertuples(index=False):
        nominal = np.flatnonzero(
            (dates >= pd.Timestamp(row.test_start))
            & (dates < pd.Timestamp(row.test_end_exclusive))
        )
        if len(nominal) <= cfg.n_horizon:
            continue
        idx = nominal[cfg.n_horizon:]
        selected = row.selected_direction
        position = (
            _position_from_signal(signals[selected].iloc[idx], cfg.n_horizon)
            if selected != "flat" else pd.Series(0.0, index=range(len(idx)))
        )
        frames.append(pd.DataFrame({
            "date": dates.iloc[idx].to_numpy(), "ret_1d": ret_1d.iloc[idx].to_numpy(),
            "position": position.to_numpy(), "window_id": row.window_id,
        }))
        transfer_log.append({
            "window_id": row.window_id, "source_index": "000905", "target_index": "000852",
            "selected_direction_from_ic": selected,
            "test_start": dates.iloc[nominal[0]], "oos_start_after_im_embargo": dates.iloc[idx[0]],
            "oos_end": dates.iloc[idx[-1]], "embargo_n_trading_days": cfg.n_horizon,
            "im_returns_used_for_selection": False,
        })
    if not frames:
        raise ValueError("IC 选向日志与 IM 历史没有可对齐的样本外窗口")
    oos = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    result = strategy.backtest_position_series(
        oos["date"], oos["position"], oos["ret_1d"], cfg.n_horizon, cfg.cost,
        position_override=oos["position"],
    )
    n_trades = sum(_window_trade_count(g["position"]) for _, g in oos.groupby("window_id"))
    output = pd.DataFrame([{
        "basis_variant": variant, "source_index": "000905", "target_index": "000852",
        "strategy": "walkforward_dynamic_ic_transfer", "n_oos_dates": len(oos),
        "n_test_windows": len(transfer_log), "exploratory": True,
        **{key: result[key] for key in METRIC_KEYS},
    }])
    output.loc[0, "n_trades"] = n_trades
    return output, pd.DataFrame(transfer_log)


def _announced_eligible(root: Path, dirname: str) -> bool:
    manifest_path = root / dirname / "run_manifest.csv"
    if not manifest_path.exists():
        return False
    manifest = pd.read_csv(manifest_path)
    return bool(manifest.get(
        "formal_backtest_eligible", pd.Series([False])
    ).fillna(False).astype(bool).all())


def run_all(output_root: str | Path) -> pd.DataFrame:
    root = Path(output_root)
    all_results = []
    all_counts = []
    raw_keys: dict[str, pd.DataFrame] = {}
    for variant, dirname in (("raw_basis", "raw_basis"), ("announced_dividend", "dividend_adjusted")):
        if variant == "announced_dividend" and not _announced_eligible(root, dirname):
            logger.warning("announced_dividend 未通过 point-in-time 资格验收，跳过。")
            continue
        summary_path = root / dirname / "summary.csv"
        if not summary_path.exists():
            logger.warning("%s 尚无结果，跳过。", variant)
            continue
        summary = pd.read_csv(summary_path, dtype={"index_code": str})
        panels: dict[str, pd.DataFrame] = {}
        variant_results = []
        variant_counts = []
        logs: dict[str, pd.DataFrame] = {}
        result_dir = root / dirname / "walkforward"
        for item in summary.itertuples(index=False):
            code = str(item.index_code).zfill(6)
            path = root / dirname / "indices" / f"{code}_{item.name}" / "factor_panel.csv"
            panel = pd.read_csv(path, parse_dates=["date"], dtype={"index_code": str})
            keys = panel[["date", "contract"]].reset_index(drop=True)
            if variant == "raw_basis":
                raw_keys[code] = keys
            elif code not in raw_keys or not keys.equals(raw_keys[code]):
                raise ValueError(f"{code} announced 与 raw 面板日期/合约不一致")
            panels[code] = panel
            result, log, counts, equity = run_walkforward(panel, code, variant)
            result.insert(2, "name", item.name)
            index_dir = result_dir / "indices" / f"{code}_{item.name}"
            index_dir.mkdir(parents=True, exist_ok=True)
            result.to_csv(index_dir / "metrics.csv", index=False)
            log.to_csv(index_dir / "selection_log.csv", index=False)
            counts.to_csv(index_dir / "selection_counts.csv", index=False)
            equity.to_csv(index_dir / "equity.csv", index=False)
            variant_results.append(result)
            all_results.append(result)
            variant_counts.append(counts)
            all_counts.append(counts)
            logs[code] = log

        if "000905" in logs and "000852" in panels:
            transfer, transfer_log = apply_ic_selection_to_im(
                panels["000852"], logs["000905"], variant
            )
            cross_dir = result_dir / "cross_asset"
            cross_dir.mkdir(parents=True, exist_ok=True)
            transfer.to_csv(cross_dir / "ic_to_im_metrics.csv", index=False)
            transfer_log.to_csv(
                cross_dir / "ic_to_im_selection_log.csv", index=False
            )
        if variant_results:
            result_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(variant_results, ignore_index=True).to_csv(
                result_dir / "summary.csv", index=False
            )
            pd.concat(variant_counts, ignore_index=True).to_csv(
                result_dir / "selection_counts_summary.csv", index=False
            )
    if not all_results:
        raise FileNotFoundError("raw/announced 均无可用面板")
    combined = pd.concat(all_results, ignore_index=True)
    comparison_dir = root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(comparison_dir / "walkforward_all_variants.csv", index=False)
    pd.concat(all_counts, ignore_index=True).to_csv(
        comparison_dir / "walkforward_selection_counts_all_variants.csv", index=False
    )
    return combined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(__file__).resolve().parent.parent / "output"
    result = run_all(root)
    display = result[[
        "basis_variant", "index_code", "name", "strategy", "n_test_windows",
        "annualized_return", "sharpe_nw", "signal_tstat_hac", "exploratory",
    ]]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
