"""neutral_hedge_timing.py —— 中性策略对冲择时（基差复盘阶段四）。

方法论性质：**规则回测**（敞口映射是可解释的规则，不是拟合出来的模型；
但状态->收敛概率的查表本身用 walk-forward 训练/测试窗口重新估计，
沿用 walkforward_direction.py 同一套 purge+embargo 纪律，避免阶段三那张
"全样本收敛概率表"里的未来信息被提前泄漏给更早的交易日）。

场景：股票多头 + 股指期货空头（用当月连续合约对冲，与 strategy.py/
walkforward_direction.py 的"当月连续"口径一致）。

规则（敞口在 [0,1] 连续映射，不做杠杆）：
    richness(t) ∈ [0,1]：把 z(t)（阶段二定义的基差偏离度）裁剪到
        [-hedge_timing_richness_z_clip, +clip] 后线性映射到 [0,1]，
        0 = 当前处于历史相对最"便宜"（深贴水）的一端，1 = 最"贵"
        （升水/贴水浅）的一端。richness 高 -> 利于开对冲（加敞口）；
        richness 低 -> 利于平对冲（减敞口）。这是敞口的基础水平。

    conviction(t) ∈ [0,1]：用 walk-forward 估计的"状态->收敛概率"表
        （只用训练窗口历史，purge 之后）查表得到 p_converge(t)，再按
        richness 所处的哪一侧决定 conviction 该看 p_converge 还是它的
        补数 (1-p_converge)：
            - richness < 0.5（便宜側）：conviction = p_converge(t)。
              p_converge 高 = 当前的"便宜"大概率很快会收敛消失，
              于是应该趁便宜、更坚决地减敞口（现在不平仓，以后更贵）。
            - richness >= 0.5（贵側）：conviction = 1 - p_converge(t)。
              p_converge 低（即"走阔"概率高）= 当前的"贵"大概率还会
              继续走阔变得更贵，于是应该趁贵、更坚决地加敞口（现在不加，
              以后开仓成本更高）。
        conviction 只是在 richness 已经给出的方向上调节"多坚决"，不会
        反转方向（0.5 是中性，既不加强也不减弱）。

    exposure(t) = clip(richness(t) + tilt_weight*(conviction(t)-0.5), 0, 1)

评价口径（重要）：不看绝对收益，只看相对"始终 100% 满额对冲"基准的
对冲成本节省，以年化基点(bps)计。单日对冲成本沿用 strategy.hedge_cost
的符号约定（基差变得更正/升水对空头对冲者不利，是正的成本）：
    daily_cost(t) = exposure(t) * ret_1d_basis(t)
    ret_1d_basis(t) 用 factors.basis_daily_return()，即单日 (t->t+1)
    基差变动占现货点位的比例，与 strategy.backtest_position_series 的
    ret_1d 口径完全一致（不是重叠窗口），避免重蹈"重叠窗口双重计数"覆辙。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from basis_model import config as config_mod
from basis_model import convergence_prob as convergence_prob_mod
from basis_model import factors as factors_mod
from basis_model import review_common as review_common_mod

logger = logging.getLogger("basis_model.neutral_hedge_timing")


def compute_richness(z: pd.Series, z_clip: float) -> pd.Series:
    """把 z 裁剪到 [-z_clip, z_clip] 后线性映射到 [0,1]（0=最便宜，1=最贵）。"""
    clipped = z.clip(-z_clip, z_clip)
    return (clipped + z_clip) / (2.0 * z_clip)


def map_exposure(
    richness: pd.Series, p_converge: pd.Series, tilt_weight: float
) -> pd.Series:
    """richness + 收敛概率的方向性 conviction 调节 -> [0,1] 连续敞口，见模块 docstring。

    direction 把 conviction 的调节方向锁定成"延伸 richness 已经指向的动作"：
    便宜侧(direction=-1)高 conviction 把敞口进一步往 0 推（更坚决地减）；
    贵侧(direction=+1)高 conviction 把敞口进一步往 1 推（更坚决地加）。
    conviction<0.5 时则是反向的"降低urgency"，敞口比纯 richness 更保守地
    留在原地，不会翻转到 richness 暗示方向的反面。
    """
    is_cheap_side = richness < 0.5
    conviction = np.where(is_cheap_side, p_converge, 1.0 - p_converge)
    direction = np.where(is_cheap_side, -1.0, 1.0)
    exposure = richness.to_numpy() + direction * tilt_weight * (conviction - 0.5)
    return pd.Series(np.clip(exposure, 0.0, 1.0), index=richness.index)


def _front_continuous_series(
    factor_panel: pd.DataFrame, cfg: config_mod.ReviewConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """构造当月连续序列 + 状态（z/z_bucket），与 walkforward_direction._front_inputs
    使用同一个"当月、按日期去重"的连续合约近似，保持全项目对"连续合约"的口径一致。
    """
    front = (
        factor_panel[factor_panel["month_tag"] == "当月"]
        .sort_values("date").drop_duplicates("date").reset_index(drop=True).copy()
    )
    state = convergence_prob_mod.attach_state(front, cfg)
    continuous = pd.Series("CONTINUOUS_FRONT", index=front.index)
    ret_1d_basis = factors_mod.basis_daily_return(
        front["adj_basis"], front["S"], continuous, front["date"]
    )
    return state, ret_1d_basis


def run_walkforward_hedge_timing(
    factor_panel: pd.DataFrame,
    index_code: str,
    variant: str,
    cfg: config_mod.ReviewConfig = config_mod.DEFAULT_REVIEW_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """purge+embargo walk-forward：每个窗口只用训练期历史重估"状态->收敛概率"表，
    再把它应用到该窗口的样本外测试期，样本外收益/成本从不参与训练期的表估计。

    Returns:
        (oos, metrics, log)
            oos: 逐日样本外敞口/成本明细。
            metrics: 汇总指标（动态 vs 基准）。
            log: 每个窗口的训练/测试边界与训练期表覆盖情况。
    """
    state, ret_1d_basis = _front_continuous_series(factor_panel, cfg)
    dates = pd.to_datetime(state["date"]).reset_index(drop=True)
    richness = compute_richness(state["z_250"].reset_index(drop=True), cfg.hedge_timing_richness_z_clip)
    z_bucket = state["z_bucket"].astype(str).reset_index(drop=True)
    ret_1d_basis = ret_1d_basis.reset_index(drop=True)

    if dates.empty:
        raise ValueError(f"{index_code} 无当月合约数据")

    purge, embargo = cfg.hedge_timing_purge_days, cfg.hedge_timing_embargo_days
    origin = dates.iloc[0] + pd.DateOffset(years=cfg.hedge_timing_train_years)
    final_date = dates.iloc[-1]
    window_frames: list[pd.DataFrame] = []
    log_rows: list[dict] = []
    window_id = 0

    while origin <= final_date:
        train_start = origin - pd.DateOffset(years=cfg.hedge_timing_train_years)
        test_end = origin + pd.DateOffset(months=cfg.hedge_timing_test_months)
        nominal_train = np.flatnonzero((dates >= train_start) & (dates < origin))
        nominal_test = np.flatnonzero((dates >= origin) & (dates < test_end))
        if len(nominal_train) > purge + cfg.n_horizon and len(nominal_test) > embargo:
            train_idx = nominal_train[:-purge] if purge > 0 else nominal_train
            oos_idx = nominal_test[embargo:]

            train_state = state.iloc[train_idx]
            train_events = convergence_prob_mod.build_convergence_events(
                train_state, n_horizons=(cfg.n_horizon,),
                std_threshold=cfg.convergence_std_threshold, decay_threshold=cfg.convergence_decay_threshold,
            )
            train_table = convergence_prob_mod.estimate_convergence_table(
                train_events, N=cfg.n_horizon, min_dte=cfg.convergence_min_dte,
                exclude_dividend_season=False, cfg=cfg,
            )
            lookup = convergence_prob_mod.bucket_lookup_table(train_table)

            test_bucket = z_bucket.iloc[oos_idx]
            p_converge = test_bucket.map(lookup)
            used_fallback = p_converge.isna()
            p_converge = p_converge.fillna(0.5)
            exposure = map_exposure(richness.iloc[oos_idx], p_converge, cfg.hedge_timing_tilt_weight)

            frame = pd.DataFrame(
                {
                    "date": dates.iloc[oos_idx].to_numpy(),
                    "window_id": window_id,
                    "z": state["z_250"].iloc[oos_idx].to_numpy(),
                    "z_bucket": test_bucket.to_numpy(),
                    "richness": richness.iloc[oos_idx].to_numpy(),
                    "p_converge": p_converge.to_numpy(),
                    "used_fallback_probability": used_fallback.to_numpy(),
                    "exposure": exposure.to_numpy(),
                    "ret_1d_basis": ret_1d_basis.iloc[oos_idx].to_numpy(),
                }
            )
            window_frames.append(frame)
            log_rows.append(
                {
                    "window_id": window_id,
                    "train_start": dates.iloc[nominal_train[0]],
                    "train_end_after_purge": dates.iloc[train_idx[-1]],
                    "purge_trading_days": purge,
                    "test_start_nominal": dates.iloc[nominal_test[0]],
                    "oos_start_after_embargo": dates.iloc[oos_idx[0]],
                    "oos_end": dates.iloc[oos_idx[-1]],
                    "embargo_trading_days": embargo,
                    "n_train_events_after_purge": int(train_table["n_obs"].sum()) if not train_table.empty else 0,
                    "n_buckets_in_train_table": int(len(train_table)),
                    "n_test_days": len(oos_idx),
                    "n_fallback_days": int(used_fallback.sum()),
                }
            )
            window_id += 1
        origin += pd.DateOffset(months=cfg.hedge_timing_test_months)

    if not window_frames:
        raise ValueError(f"{index_code} 历史不足以形成 {cfg.hedge_timing_train_years}年训练/"
                          f"{cfg.hedge_timing_test_months}月测试窗口")

    oos = pd.concat(window_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if oos["date"].duplicated().any():
        raise AssertionError("walk-forward 测试窗口发生重叠")
    log = pd.DataFrame(log_rows)

    oos["dynamic_daily_cost"] = oos["exposure"] * oos["ret_1d_basis"]
    oos["benchmark_daily_cost"] = oos["ret_1d_basis"]  # 始终 100% 满额对冲
    oos["daily_cost_saving"] = oos["benchmark_daily_cost"] - oos["dynamic_daily_cost"]

    metrics = _summarize_cost_saving(oos, index_code, variant, cfg)
    return oos, metrics, log


def _hac_tstat(series: pd.Series, lags: int) -> float:
    clean = series.dropna()
    n = len(clean)
    if n <= lags + 2 or clean.std(ddof=1) == 0:
        return float("nan")
    X = np.ones((n, 1))
    model = sm.OLS(clean.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.tvalues[0])


def _summarize_cost_saving(
    oos: pd.DataFrame, index_code: str, variant: str, cfg: config_mod.ReviewConfig
) -> pd.DataFrame:
    n_days = len(oos)
    benchmark_annualized_bps = float(oos["benchmark_daily_cost"].mean() * 252 * 10_000)
    dynamic_annualized_bps = float(oos["dynamic_daily_cost"].mean() * 252 * 10_000)
    saving_bps = benchmark_annualized_bps - dynamic_annualized_bps
    turnover = float(oos["exposure"].diff().abs().fillna(0.0).mean())
    return pd.DataFrame(
        [
            {
                "basis_variant": variant,
                "index_code": index_code,
                "n_oos_days": n_days,
                "n_windows": int(oos["window_id"].nunique()),
                "mean_exposure": float(oos["exposure"].mean()),
                "mean_daily_turnover": turnover,
                "benchmark_annualized_cost_bps": benchmark_annualized_bps,
                "dynamic_annualized_cost_bps": dynamic_annualized_bps,
                "cost_saving_annualized_bps": saving_bps,
                "cost_saving_tstat_hac": _hac_tstat(oos["daily_cost_saving"], cfg.n_horizon),
                "fallback_probability_share": float(oos["used_fallback_probability"].mean()),
                "purge_days": cfg.hedge_timing_purge_days,
                "embargo_days": cfg.hedge_timing_embargo_days,
            }
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = config_mod.DEFAULT_REVIEW_CONFIG
    items = review_common_mod.discover_variant_indices()
    if not items:
        logger.warning("output/ 下没有任何 factor_panel.csv，先跑 run.py 再执行本模块。")
        return

    summaries = []
    for item in items:
        panel = review_common_mod.load_factor_panel(item.factor_panel_path)
        try:
            oos, metrics, log = run_walkforward_hedge_timing(panel, item.index_code, item.variant, cfg)
        except ValueError as exc:
            logger.warning("[%s/%s] 跳过：%s", item.variant, item.name, exc)
            continue

        out_dir = review_common_mod.review_output_dir(
            review_common_mod.OUTPUT_ROOT, item.variant, cfg.output_subdir
        )
        oos.to_csv(out_dir / f"hedge_timing_{item.index_code}.csv", index=False)
        log.to_csv(out_dir / f"hedge_timing_windows_{item.index_code}.csv", index=False)

        metrics = metrics.copy()
        metrics.insert(1, "name", item.name)
        summaries.append(metrics)

        row = metrics.iloc[0]
        logger.info(
            "[%s/%s] 动态敞口 vs 满额对冲基准：年化成本节省 %.2f bps（HAC t=%.2f），"
            "平均敞口 %.2f，样本外 %d 天，%d 个窗口",
            item.variant, item.name, row["cost_saving_annualized_bps"], row["cost_saving_tstat_hac"],
            row["mean_exposure"], row["n_oos_days"], row["n_windows"],
        )

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        for variant, grp in combined.groupby("basis_variant"):
            out_dir = review_common_mod.review_output_dir(
                review_common_mod.OUTPUT_ROOT, variant, cfg.output_subdir
            )
            grp.to_csv(out_dir / "cost_saving_summary.csv", index=False)
    else:
        logger.warning("没有任何指数产出足够的 walk-forward 样本外窗口。")


if __name__ == "__main__":
    main()
