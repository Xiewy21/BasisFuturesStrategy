"""run.py —— 十年期理论回测：端到端跑通 data -> dividend -> factors/regression
-> strategy -> forecast 四层，参数全部来自 config.py。

【口径声明】这是一次"理论回测"：区间为运行日向前滚动十年，
无摩擦成本（config.COST=0），不对照任何研究报告的历史数字，只用统一规则
观察策略/因子在十年尺度上是否稳定。唯一的主回测方法是
strategy.backtest_position_series()（Prompt A 修好的仓位序列法）；
strategy.backtest_naive_overlapping() 禁止用于本文件的主结果，只在 README 里
作为"重叠窗口双重计数"的反面教材保留。

默认行为：
    - 数据源：默认中金所官方月度期货行情 + AkShare 现货指数。真实取数失败时
      默认终止该指数，不会把 MockAdapter 模拟数据伪装成十年回测。
      只有显式设置 allow_mock_backtest=True 时才允许模拟演示。
    - 分红：当前默认运行原始基差（assume_zero_dividend=True），
      结果独立保存到 output/raw_basis，不会被后续本地分红修正版覆盖。
    - 每个指数按 config.INDEX_CONFIGS 里各自的 data_start 独立跑，任何一个
      指数出错都会被捕获、记录日志后跳过，不会中断整轮回测。
    - 输出：每个指数产出因子回归系数(含 Newey-West 显著性)、择时净值曲线与指标、
      对冲成本阶梯、敏感度表、多 N 稳健性对照，共 7 张 CSV；最后额外产出一张
      跨指数汇总表 output/raw_basis/summary.csv。

这是一份"研究脚本"，不是服务/守护进程，也不做任何下单/实盘对接。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from basis_model import config as config_mod
from basis_model import data as data_mod
from basis_model import dividend as dividend_mod
from basis_model import factors as factors_mod
from basis_model import forecast as forecast_mod
from basis_model import regression as regression_mod
from basis_model import strategy as strategy_mod

logger = logging.getLogger("basis_model.run")


# --------------------------------------------------------------------------- #
# 数据源解析 / 分红多年份拉取
# --------------------------------------------------------------------------- #

def resolve_primary_adapter(config: config_mod.BacktestConfig) -> data_mod.DataAdapter:
    """按 config.data_source_primary 解析主数据源；WindAdapter 目前是桩实现，
    检测到就直接回退到 config.data_source_fallback 并打印告警，不浪费一次
    必然失败的真实调用。
    """
    primary = data_mod.get_adapter(config.data_source_primary, cache_dir=str(config.cache_dir))
    if isinstance(primary, data_mod.WindAdapter):
        logger.warning(
            "数据源 %r (WindAdapter) 目前是桩实现，尚未接入 WindPy，自动回退到 %r。",
            config.data_source_primary,
            config.data_source_fallback,
        )
        primary = data_mod.get_adapter(config.data_source_fallback, cache_dir=str(config.cache_dir))
    return primary


def load_dividend_events_for_range(
    adapter: data_mod.DataAdapter, index_code: str, start_date: str, end_date: str
) -> Optional[pd.DataFrame]:
    """按年份逐年调用 adapter.load_dividends 并拼接成覆盖整个回测区间的分红事件表。

    单年失败/为空不影响其他年份；全部年份都失败才返回 None（触发调用方回退
    assume_zero_dividend=True）。
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    frames: list[pd.DataFrame] = []
    failed_years: list[int] = []
    for year in range(start_year, end_year + 1):
        try:
            year_df = adapter.load_dividends(index_code, year)
        except Exception as exc:
            logger.debug("%s %d 年分红数据获取异常: %r", index_code, year, exc)
            failed_years.append(year)
            continue
        if year_df is None or year_df.empty:
            failed_years.append(year)
            continue
        frames.append(year_df)

    if failed_years:
        logger.info(
            "%s 有 %d/%d 个年份未取到分红数据（%s），已跳过这些年份继续拼接。",
            index_code,
            len(failed_years),
            end_year - start_year + 1,
            failed_years,
        )
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 单指数流水线：下面 6 个 _xxx_layer/_write_outputs/_build_summary_row 函数
# 一一对应 README 的四层流水线（data -> dividend -> factors+regression ->
# strategy -> forecast）加落盘/汇总；run_for_index() 只负责按顺序编排调用，
# 不包含具体计算逻辑，方便按层定位代码。
# --------------------------------------------------------------------------- #

def _load_data_layer(
    index_cfg: config_mod.IndexConfig,
    primary_adapter: data_mod.DataAdapter,
    mock_adapter: Optional[data_mod.MockAdapter],
    start_date: str,
    end_date: str,
) -> Optional[tuple[pd.DataFrame, str]]:
    """数据层：取期货+现货合并面板。面板为空时返回 None，调用方据此跳过该指数。"""
    t_data = time.perf_counter()
    panel = data_mod.load_panel_with_fallback(
        primary=primary_adapter,
        fallback=mock_adapter,
        index_code=index_cfg.index_code,
        start_date=start_date,
        end_date=end_date,
    )
    data_source_used = panel.attrs.get("data_source", type(primary_adapter).__name__)
    logger.info(
        "[%s] 数据层：面板 %d 行，合约数 %d，耗时 %.1fs",
        index_cfg.name,
        len(panel),
        panel["contract"].nunique() if not panel.empty else 0,
        time.perf_counter() - t_data,
    )
    if panel.empty:
        logger.warning("[%s] 无可用数据，整个指数跳过。", index_cfg.name)
        return None
    return panel, data_source_used


def _dividend_layer(
    panel: pd.DataFrame,
    index_cfg: config_mod.IndexConfig,
    config: config_mod.BacktestConfig,
    primary_adapter: data_mod.DataAdapter,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, bool]:
    """分红层（结构建模，顺序刚性纪律要求剔分红优先于造因子）。

    Returns:
        (分红调整后的面板, 本指数实际生效的 assume_zero_dividend)。
    """
    t_div = time.perf_counter()
    assume_zero_dividend = config.assume_zero_dividend_default
    dividend_events = None
    if not assume_zero_dividend:
        dividend_events = load_dividend_events_for_range(
            primary_adapter, index_cfg.index_code, start_date, end_date
        )
        if dividend_events is None or dividend_events.empty:
            assume_zero_dividend = True
            logger.warning(
                "[%s] 分红季基差未剔除：数据源拿不到历史权重/分红点位，"
                "本指数自动回退 assume_zero_dividend=True。",
                index_cfg.name,
            )
    panel_adj = dividend_mod.dividend_adjust_panel(
        panel, dividend_events=dividend_events, assume_zero_dividend=assume_zero_dividend
    )
    logger.info("[%s] 分红层：耗时 %.1fs", index_cfg.name, time.perf_counter() - t_div)
    return panel_adj, assume_zero_dividend


def _factor_regression_layer(
    panel_adj: pd.DataFrame,
    index_cfg: config_mod.IndexConfig,
    config: config_mod.BacktestConfig,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """因子层 + 回归层（统计建模）：对 config.all_n_values 逐个 N 做稳健性扫描；
    主 N（config.n_horizon）的因子面板/系数额外保留给策略层/预测层使用。

    Returns:
        (稳健性对照表, 主 N 因子面板 或 None, 主 N 回归系数 或 None)。
    """
    t_robust = time.perf_counter()
    robustness_rows = []
    primary_factor_panel: Optional[pd.DataFrame] = None
    primary_coefs: Optional[pd.DataFrame] = None

    for n_value in config.all_n_values:
        factor_panel_n = factors_mod.build_factor_panel(panel_adj, N=n_value)
        factor_panel_n = factor_panel_n.sort_values(["contract", "date"]).reset_index(drop=True)
        coefs_n = regression_mod.rolling_ols(
            factor_panel_n,
            window=config.rolling_window,
            min_obs=config.min_obs,
            step=config.refit_every,
            hac_lags=n_value,
        )

        if n_value == config.n_horizon:
            primary_factor_panel = factor_panel_n
            primary_coefs = coefs_n

        if coefs_n.empty:
            robustness_rows.append(
                {
                    "N": n_value,
                    "n_periods": 0,
                    "a1_mean": float("nan"),
                    "a1_latest": float("nan"),
                    "a1_tstat_latest": float("nan"),
                    "a1_significant_share": float("nan"),
                    "r_squared_mean": float("nan"),
                }
            )
            continue

        a1_col = "f1_ann_basis"
        tstat_col = f"tstat_{a1_col}"
        a1_significant_share = float("nan")
        a1_tstat_latest = float("nan")
        if tstat_col in coefs_n.columns:
            a1_significant_share = float((coefs_n[tstat_col].abs() >= 1.96).mean())
            a1_tstat_latest = float(coefs_n[tstat_col].iloc[-1])

        robustness_rows.append(
            {
                "N": n_value,
                "n_periods": len(coefs_n),
                "a1_mean": float(coefs_n[a1_col].mean()) if a1_col in coefs_n.columns else float("nan"),
                "a1_latest": float(coefs_n[a1_col].iloc[-1]) if a1_col in coefs_n.columns else float("nan"),
                "a1_tstat_latest": a1_tstat_latest,
                "a1_significant_share": a1_significant_share,
                "r_squared_mean": float(coefs_n["r_squared"].mean()),
            }
        )

    robustness_df = pd.DataFrame(robustness_rows)
    logger.info(
        "[%s] 因子/回归层：完成 N=%s 稳健性扫描，耗时 %.1fs",
        index_cfg.name,
        config.all_n_values,
        time.perf_counter() - t_robust,
    )
    return robustness_df, primary_factor_panel, primary_coefs


def _forward_fill_daily_coefs(factor_panel: pd.DataFrame, coefs: pd.DataFrame) -> pd.DataFrame:
    """把 walk-forward 系数（每 refit_every 天才重估一次）前向填充到每个交易日，
    供择时展期(hedge_cost_ladder)与预测层逐日使用；不产生前视，因为 ffill
    只会把"过去某天估出的系数"沿用到之后没有重估的日子，不会用到未来系数。
    """
    all_dates = pd.Index(sorted(factor_panel["date"].unique()), name="date")
    return coefs.reindex(all_dates).ffill()


def _strategy_layer(
    factor_panel: pd.DataFrame,
    coefs: pd.DataFrame,
    coefs_daily: pd.DataFrame,
    index_cfg: config_mod.IndexConfig,
    config: config_mod.BacktestConfig,
) -> tuple[dict, pd.DataFrame]:
    """策略层（规则回测，仓位序列法为唯一主方法）：择时回测 + 对冲成本阶梯。

    就地给 factor_panel 加一列 hedge_cost（调用方持有的同一个 DataFrame 对象
    会被修改，与原实现行为一致，不做防御性拷贝）。
    """
    t_strat = time.perf_counter()
    factor_panel["hedge_cost"] = strategy_mod.hedge_cost(
        factor_panel["adj_basis"], factor_panel["contract"], factor_panel["date"], config.n_horizon
    )
    # 主择时回测必须每日一条。旧实现把同日四个合约当成四个
    # 连续时点复利，会破坏年化口径。这里使用「连续近月」规则，
    # 合约代码统一成连续序列标识，使分位数窗口真正跨合约滚动。
    timing_panel = (
        factor_panel[factor_panel["month_tag"] == "当月"]
        .sort_values("date")
        .drop_duplicates("date", keep="first")
        .copy()
    )
    continuous_contract = pd.Series("CONTINUOUS_FRONT", index=timing_panel.index)
    timing_panel["timing_signal"] = strategy_mod.map_by_index(
        index_cfg.index_code,
        timing_panel["f1_ann_basis"],
        continuous_contract,
        timing_panel["date"],
        window=config.timing_window,
        pct=config.timing_pct,
    )
    timing_panel["ret_1d"] = factors_mod.basis_daily_return(
        timing_panel["adj_basis"],
        timing_panel["S"],
        continuous_contract,
        timing_panel["date"],
    )

    backtest_result = strategy_mod.backtest_position_series(
        dates=timing_panel["date"],
        signal=timing_panel["timing_signal"],
        ret_1d=timing_panel["ret_1d"],
        N=config.n_horizon,
        cost=config.cost,
    )
    logger.info(
        "[%s] 择时规则回测（仓位序列法）：年化收益 %.2f%%, 年化波动 %.2f%%, Sharpe %.2f, "
        "Sharpe(NW) %.2f, 最大回撤 %.2f%%, 胜率 %s, 换仓 %d 次, HAC t值 %.2f",
        index_cfg.name,
        backtest_result["annualized_return"] * 100,
        backtest_result["annualized_vol"] * 100,
        backtest_result["sharpe"],
        backtest_result["sharpe_nw"],
        backtest_result["max_drawdown"] * 100,
        f"{backtest_result['win_rate']:.2%}" if pd.notna(backtest_result["win_rate"]) else "N/A",
        backtest_result["n_trades"],
        backtest_result["signal_tstat_hac"],
    )

    hedge_ladder = strategy_mod.hedge_cost_ladder(factor_panel, coefs_daily, N=config.n_horizon)
    if not hedge_ladder.empty:
        best_hedge = hedge_ladder.iloc[0]
        logger.info(
            "[%s] 空头展仓成本：最低规则=%s，平均年化成本=%.4f%%（完整表见输出 CSV）",
            index_cfg.name,
            best_hedge["策略"],
            best_hedge["mean_annualized_cost"] * 100,
        )
    logger.info("[%s] 策略层：耗时 %.1fs", index_cfg.name, time.perf_counter() - t_strat)
    return backtest_result, hedge_ladder


def _forecast_layer(
    factor_panel: pd.DataFrame,
    coefs: pd.DataFrame,
    index_cfg: config_mod.IndexConfig,
    config: config_mod.BacktestConfig,
) -> pd.DataFrame:
    """预测层（硬:回归外推 + 软:定性叠加）：取最新交易日的当月/次月合约产出情景敏感度表。"""
    t_forecast = time.perf_counter()
    sensitivity_df = pd.DataFrame()
    if not coefs.empty:
        latest_date = factor_panel["date"].max()
        latest_rows = factor_panel[factor_panel["date"] == latest_date]
        target_row = None
        for tag in ("当月", "次月"):
            tagged = latest_rows[latest_rows["month_tag"] == tag]
            if not tagged.empty:
                target_row = tagged.iloc[0]
                break
        if target_row is None and not latest_rows.empty:
            target_row = latest_rows.iloc[0]

        if target_row is not None:
            coef_row = coefs.iloc[-1]
            month_dummy_cols = [c for c in factor_panel.columns if c.startswith("D_month_")]
            month_dummy_values = {c: float(target_row[c]) for c in month_dummy_cols}

            sensitivity_df = forecast_mod.sensitivity_table(
                date=target_row["date"],
                contract=target_row["contract"],
                index_code=index_cfg.index_code,
                S=float(target_row["S"]),
                current_adj_basis=float(target_row["adj_basis"]),
                f1_ann_basis=float(target_row["f1_ann_basis"]),
                f3_term_structure=float(target_row["f3_term_structure"])
                if pd.notna(target_row["f3_term_structure"])
                else 0.0,
                coef_row=coef_row,
                N=config.n_horizon,
                expected_returns=config.sensitivity_expected_returns,
                month_tag=str(target_row["month_tag"]),
                month_dummy_values=month_dummy_values,
            )
    logger.info("[%s] 预测层：耗时 %.1fs", index_cfg.name, time.perf_counter() - t_forecast)
    return sensitivity_df


def _write_outputs(
    config: config_mod.BacktestConfig,
    index_cfg: config_mod.IndexConfig,
    factor_panel: pd.DataFrame,
    coefs: pd.DataFrame,
    backtest_result: dict,
    hedge_ladder: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> None:
    """落盘：总览留在根目录，每个指数的明细放进独立子目录。"""
    index_dir = config.output_dir / "indices" / f"{index_cfg.index_code}_{index_cfg.name}"
    index_dir.mkdir(parents=True, exist_ok=True)
    factor_panel.to_csv(index_dir / "factor_panel.csv", index=False)
    coefs.to_csv(index_dir / "regression_coefs.csv")
    backtest_result["equity_curve"].to_csv(index_dir / "backtest_equity.csv", index=False)
    pd.DataFrame(
        [{k: v for k, v in backtest_result.items() if k != "equity_curve"}]
    ).to_csv(index_dir / "timing_metrics.csv", index=False)
    hedge_ladder.to_csv(index_dir / "hedge_cost_ladder.csv", index=False)
    hedge_ladder.attrs.get("roll_events", pd.DataFrame()).to_csv(
        index_dir / "hedge_roll_events.csv", index=False
    )
    # 空结果也覆盖写入，避免上一次运行的敏感度表残留并冒充本轮结果。
    sensitivity_df.to_csv(index_dir / "sensitivity_table.csv", index=False)
    robustness_df.to_csv(index_dir / "robustness.csv", index=False)
    logger.info("[%s] 明细输出：%s", index_cfg.name, index_dir)


def _build_summary_row(
    index_cfg: config_mod.IndexConfig,
    start_date: str,
    factor_panel: pd.DataFrame,
    coefs: pd.DataFrame,
    backtest_result: dict,
    hedge_ladder: pd.DataFrame,
    data_source_used: str,
    assume_zero_dividend: bool,
) -> dict:
    """跨指数汇总表 summary.csv 里该指数对应的一行。"""
    a1_col = "f1_ann_basis"
    tstat_col = f"tstat_{a1_col}"
    a1_mean = float(coefs[a1_col].mean()) if a1_col in coefs.columns and not coefs.empty else float("nan")
    a1_significant_share = float("nan")
    if tstat_col in coefs.columns and not coefs.empty:
        a1_significant_share = float((coefs[tstat_col].abs() >= 1.96).mean())

    hedge_by_rule = dict(zip(hedge_ladder["策略"], hedge_ladder["mean_annualized_cost"]))

    return {
        "index_code": index_cfg.index_code,
        "name": index_cfg.name,
        "data_source": data_source_used,
        "data_start": start_date,
        "actual_data_start": factor_panel["date"].min(),
        "actual_data_end": factor_panel["date"].max(),
        "n_trading_days": factor_panel["date"].nunique(),
        "front_month_coverage": (
            factor_panel.loc[factor_panel["month_tag"] == "当月", "date"].nunique()
            / max(factor_panel["date"].nunique(), 1)
        ),
        "short_history": index_cfg.short_history_note is not None,
        "assume_zero_dividend_used": assume_zero_dividend,
        "basis_variant": "raw_basis" if assume_zero_dividend else "dividend_adjusted",
        "n_panel_rows": len(factor_panel),
        "annualized_return": backtest_result["annualized_return"],
        "annualized_vol": backtest_result["annualized_vol"],
        "sharpe": backtest_result["sharpe"],
        "sharpe_nw": backtest_result["sharpe_nw"],
        "max_drawdown": backtest_result["max_drawdown"],
        "win_rate": backtest_result["win_rate"],
        "n_trades": backtest_result["n_trades"],
        "signal_tstat_hac": backtest_result["signal_tstat_hac"],
        "a1_mean": a1_mean,
        "a1_significant_share": a1_significant_share,
        "hedge_cost_当月": hedge_by_rule.get("当月", float("nan")),
        "hedge_cost_季月": hedge_by_rule.get("季月", float("nan")),
        "hedge_cost_最低贴水": hedge_by_rule.get("最低贴水", float("nan")),
        "hedge_cost_回归展仓": hedge_by_rule.get("回归展仓", float("nan")),
    }


def run_for_index(
    index_cfg: config_mod.IndexConfig,
    config: config_mod.BacktestConfig,
    primary_adapter: data_mod.DataAdapter,
    mock_adapter: Optional[data_mod.MockAdapter],
) -> dict:
    """对单一指数跑完四层流水线 + 多 N 稳健性扫描，返回汇总行与关键中间结果。

    本函数只负责按 data -> dividend -> factors/regression -> strategy ->
    forecast 的顺序编排调用上面的 _xxx_layer 函数，不包含具体计算逻辑；任何
    一步失败都应该由调用方（main）捕获，本函数只负责单一指数内部"缺数据自动
    跳过、不因局部缺失而整体崩溃"。
    """
    t0 = time.perf_counter()
    start_date = index_cfg.data_start
    end_date = config.end_date
    logger.info("[%s] 开始处理，区间 %s ~ %s", index_cfg.name, start_date, end_date)
    if index_cfg.short_history_note:
        logger.warning("[%s] %s", index_cfg.name, index_cfg.short_history_note)

    loaded = _load_data_layer(index_cfg, primary_adapter, mock_adapter, start_date, end_date)
    if loaded is None:
        return {}
    panel, data_source_used = loaded

    panel_adj, assume_zero_dividend = _dividend_layer(
        panel, index_cfg, config, primary_adapter, start_date, end_date
    )

    robustness_df, factor_panel, coefs = _factor_regression_layer(panel_adj, index_cfg, config)
    if factor_panel is None or coefs is None:
        logger.warning("[%s] 主 N=%d 未产出因子面板，跳过策略/预测层。", index_cfg.name, config.n_horizon)
        return {}

    coefs_daily = _forward_fill_daily_coefs(factor_panel, coefs)
    backtest_result, hedge_ladder = _strategy_layer(factor_panel, coefs, coefs_daily, index_cfg, config)
    sensitivity_df = _forecast_layer(factor_panel, coefs, index_cfg, config)

    _write_outputs(
        config, index_cfg, factor_panel, coefs, backtest_result, hedge_ladder, sensitivity_df, robustness_df
    )
    summary_row = _build_summary_row(
        index_cfg, start_date, factor_panel, coefs, backtest_result, hedge_ladder,
        data_source_used, assume_zero_dividend,
    )

    logger.info("[%s] 处理完成，总耗时 %.1fs", index_cfg.name, time.perf_counter() - t0)

    return {
        "summary_row": summary_row,
        "factor_panel": factor_panel,
        "coefs": coefs,
        "backtest_result": backtest_result,
        "hedge_ladder": hedge_ladder,
        "sensitivity_df": sensitivity_df,
        "robustness_df": robustness_df,
    }


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = config_mod.DEFAULT_CONFIG
    logger.info(
        "basis_model.run —— 十年期理论回测（研究/回测工具，非实盘系统）：区间 %s ~ %s, "
        "主数据源 %r, N=%d, 稳健性 N=%s, cost=%s",
        config.start_date,
        config.end_date,
        config.data_source_primary,
        config.n_horizon,
        config.n_robustness,
        config.cost,
    )

    primary_adapter = resolve_primary_adapter(config)
    mock_adapter = data_mod.get_adapter("mock") if config.allow_mock_backtest else None

    summary_rows = []
    for index_cfg in config.index_configs:
        try:
            result = run_for_index(index_cfg, config, primary_adapter, mock_adapter)
        except Exception:
            logger.exception("[%s] 处理过程中出现未预期异常，跳过该指数，继续下一个。", index_cfg.name)
            continue
        if result:
            summary_rows.append(result["summary_row"])

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(config.output_dir / "summary.csv", index=False)
        # 为后续本地 point-in-time 分红修正版保留可机器读取的口径声明。
        manifest = pd.DataFrame(
            [
                {
                    "basis_variant": "raw_basis"
                    if config.assume_zero_dividend_default
                    else "dividend_adjusted",
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "futures_source": type(primary_adapter).__name__,
                    "dividend_adjustment": "none"
                    if config.assume_zero_dividend_default
                    else "point_in_time_required",
                }
            ]
        )
        manifest.to_csv(config.output_dir / "run_manifest.csv", index=False)
        display_cols = [
            "index_code", "name", "annualized_return", "sharpe_nw",
            "max_drawdown", "n_trades",
        ]
        logger.info("跨指数关键结果：\n%s", summary_df[display_cols].to_string(index=False))
        logger.info("完整跨指数汇总：%s", config.output_dir / "summary.csv")
    else:
        logger.warning("没有任何指数产出结果，未生成 summary.csv。")

    logger.info("完成。输出已保存至: %s", config.output_dir)


if __name__ == "__main__":
    main()
