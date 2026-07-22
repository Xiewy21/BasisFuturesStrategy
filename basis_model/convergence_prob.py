"""convergence_prob.py —— 收敛概率（基差复盘阶段三）。

方法论性质：**统计建模**（经验事件研究：给定状态分桶，未来 N 日内是否发生
"收敛"是一个历史频率统计，不是可实时更新的预测模型）。

【口径声明——这是历史经验统计，不是样本外预测】
    estimate_convergence_table() 用传入的整段历史一次性估计"状态 -> 收敛概率"，
    这段历史包含了"未来"（相对于历史中较早的观测点而言）。这对"复盘、回答
    历史上某类状态平均收敛得有多快"这个研究问题是合适的做法，类似任何事件
    研究(event study)都是用全样本估计条件概率分布。但如果要把这张表当成
    实盘每日更新的交易信号，必须只用截止到当天为止的历史重新估计（walk-forward），
    否则会把"未来才知道的收敛结果"提前泄漏给更早的交易日。neutral_hedge_timing.py
    正是这样做的：它按 purge+embargo 把历史切成训练/测试窗口，每个窗口只用训练期
    数据调用本模块的 estimate_convergence_table()，再应用到该窗口的测试期，
    不会把测试期已经发生的收敛结果算进训练用的概率表里。

必须处理的三个陷阱（详见 build_convergence_events docstring）：
    (a) 到期效应：临近交割的基差会机械收敛，必须按 days_to_expiry 分层。
    (b) 重叠样本：未来 N 日窗口逐日滚动天然重叠，标准 OLS/二项分布标准误会
        低估真实不确定性，用 block bootstrap 给出置信区间。
    (c) 分红污染：raw 口径未剔分红，5-7月的"收敛"可能只是除息造成的假象。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from basis_model import basis_center as basis_center_mod
from basis_model import config as config_mod
from basis_model import dividend as dividend_mod
from basis_model import review_common as review_common_mod

logger = logging.getLogger("basis_model.convergence_prob")


# --------------------------------------------------------------------------- #
# 状态变量：z 分桶
# --------------------------------------------------------------------------- #

def bucket_labels(edges: tuple[float, ...] = config_mod.CONVERGENCE_Z_BUCKET_EDGES) -> list[str]:
    """按 z 由小到大排好序的分桶标签列表，供分桶与画图共用同一份顺序定义。

    调用方（尤其是画图代码）不能依赖 pandas 对分桶结果 .to_numpy() 之后的
    普通字符串数组重新排序 —— 那会退化成字典序（"(-1,0]" 排在 "(-2,-1]"
    前面），必须显式用这份按数值大小排好的顺序。
    """
    bins = [-np.inf, *edges, np.inf]
    labels = [f"({bins[i]:.0f},{bins[i+1]:.0f}]" for i in range(len(bins) - 1)]
    labels[0] = f"(-inf,{bins[1]:.0f}]"
    labels[-1] = f"({bins[-2]:.0f},inf)"
    return labels


def label_zscore_bucket(
    z: pd.Series, edges: tuple[float, ...] = config_mod.CONVERGENCE_Z_BUCKET_EDGES
) -> pd.Series:
    """按 z 分桶，区间左开右闭，两端补 ±inf。"""
    bins = [-np.inf, *edges, np.inf]
    labels = bucket_labels(edges)
    return pd.cut(z, bins=bins, labels=labels, include_lowest=True)


def attach_state(
    factor_panel: pd.DataFrame,
    cfg: config_mod.ReviewConfig = config_mod.DEFAULT_REVIEW_CONFIG,
) -> pd.DataFrame:
    """给 factor_panel 附加因果滚动 center/std/z/bucket 四列（沿用阶段二的实现）。

    只调用 basis_center 的纯函数，不读取阶段二落盘的 CSV，因此本模块可以
    独立于 basis_center.py 是否已经跑过而运行，也让 neutral_hedge_timing.py
    可以对任意子集（训练窗口）重复调用本函数而不依赖磁盘状态。
    """
    stats = basis_center_mod.causal_rolling_stats(
        factor_panel["f1_ann_basis"], factor_panel["index_code"], factor_panel["month_tag"],
        factor_panel["date"], window=cfg.basis_center_window, min_periods=cfg.basis_center_min_periods,
    )
    z = basis_center_mod.compute_z_score(factor_panel["f1_ann_basis"], stats["center"], stats["std"])
    out = factor_panel.copy()
    out["center_250"] = stats["center"]
    out["std_250"] = stats["std"]
    out["z_250"] = z
    out["z_bucket"] = label_zscore_bucket(z, cfg.convergence_z_bucket_edges)
    return out


# --------------------------------------------------------------------------- #
# 收敛事件构造
# --------------------------------------------------------------------------- #

def _events_for_group(
    f1: np.ndarray,
    center: np.ndarray,
    std: np.ndarray,
    dates: np.ndarray,
    N: int,
    std_threshold: float,
    decay_threshold: float,
) -> pd.DataFrame:
    """单个 (index_code, month_tag) 分组、按日期排序后的收敛事件（向量化外层循环，
    内层最多 N 步，N<=60 时对 ~2500 个交易日完全可接受）。

    收敛判定：未来 [t+1, t+N] 内第一次出现 |f1_k - center_t| < std_threshold*std_t
    或相对初始偏离收敛掉 >= decay_threshold 比例，即记为收敛，记录用了几天。
    历史最后不足 N 个交易日的观测视为"删失"（censored）—— 无法确认是否会在
    完整 N 日内收敛，标记 converged=NaN 而不是 False，避免把"数据不够长"
    误判成"没有收敛"，压低经验收敛概率（与 factors.py 对 f2/y 最后 N 行
    统一标 NaN 是同一纪律）。
    """
    n = len(f1)
    converged = np.full(n, np.nan)
    days_to_converge = np.full(n, np.nan)
    reduction_frac = np.full(n, np.nan)
    touches_dividend = np.zeros(n, dtype=bool)
    is_dividend_season = np.array([dividend_mod.is_dividend_season(pd.Timestamp(d)) for d in dates])

    for t in range(n):
        c, s = center[t], std[t]
        if np.isnan(c) or np.isnan(s) or s <= 0 or np.isnan(f1[t]):
            continue
        if t + N >= n:
            continue  # 删失：历史不够长，无法确认完整 N 日窗口内是否收敛
        touches_dividend[t] = bool(is_dividend_season[t : t + N + 1].any())
        initial_dev = abs(f1[t] - c)
        if initial_dev == 0:
            converged[t], days_to_converge[t], reduction_frac[t] = True, 0.0, 1.0
            continue
        best_reduction = -np.inf
        found_at = None
        for k in range(t + 1, t + N + 1):
            if np.isnan(f1[k]):
                continue
            dev_k = abs(f1[k] - c)
            reduction = 1.0 - dev_k / initial_dev
            best_reduction = max(best_reduction, reduction)
            if dev_k < std_threshold * s or reduction >= decay_threshold:
                found_at = k
                break
        if found_at is not None:
            converged[t] = True
            days_to_converge[t] = found_at - t
            reduction_frac[t] = 1.0 - abs(f1[found_at] - c) / initial_dev
        else:
            converged[t] = False
            reduction_frac[t] = best_reduction if best_reduction > -np.inf else np.nan

    return pd.DataFrame(
        {
            "converged": converged,
            "days_to_converge": days_to_converge,
            "reduction_frac": reduction_frac,
            "touches_dividend_season": touches_dividend,
        }
    )


def build_convergence_events(
    state_panel: pd.DataFrame,
    n_horizons: tuple[int, ...] = config_mod.CONVERGENCE_N_HORIZONS,
    std_threshold: float = config_mod.CONVERGENCE_STD_THRESHOLD,
    decay_threshold: float = config_mod.CONVERGENCE_DECAY_THRESHOLD,
) -> pd.DataFrame:
    """对每个 (index_code, month_tag, N) 组合构造收敛事件，长表输出。

    Args:
        state_panel: attach_state() 的输出（含 center_250/std_250/z_250/z_bucket）。

    Returns:
        长表，一行 = 一个 (index_code, month_tag, date, N) 观测，列：
        z, z_bucket, days_to_expiry, converged, days_to_converge, reduction_frac,
        touches_dividend_season。
    """
    frames = []
    for (code, tag), grp in state_panel.groupby(["index_code", "month_tag"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        f1 = grp["f1_ann_basis"].to_numpy()
        center = grp["center_250"].to_numpy()
        std = grp["std_250"].to_numpy()
        dates = grp["date"].to_numpy()
        for N in n_horizons:
            events = _events_for_group(f1, center, std, dates, N, std_threshold, decay_threshold)
            block = pd.DataFrame(
                {
                    "index_code": code,
                    "month_tag": tag,
                    "date": grp["date"].to_numpy(),
                    "N": N,
                    "z": grp["z_250"].to_numpy(),
                    "z_bucket": grp["z_bucket"].to_numpy(),
                    "days_to_expiry": grp["days_to_expiry"].to_numpy(),
                }
            )
            frames.append(pd.concat([block, events], axis=1))
    if not frames:
        return pd.DataFrame(
            columns=[
                "index_code", "month_tag", "date", "N", "z", "z_bucket", "days_to_expiry",
                "converged", "days_to_converge", "reduction_frac", "touches_dividend_season",
            ]
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# block bootstrap 置信区间（应对重叠样本低估标准误）
# --------------------------------------------------------------------------- #

def block_bootstrap_ci(
    values: np.ndarray,
    block_len: int,
    reps: int = config_mod.CONVERGENCE_BOOTSTRAP_REPS,
    ci: tuple[float, float] = config_mod.CONVERGENCE_BOOTSTRAP_CI,
    seed: int = config_mod.CONVERGENCE_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """moving block bootstrap：按时间顺序重叠分块重采样，估计均值的置信区间。

    values 必须已经按时间排序（本模块调用方保证）。block_len 一般取事件本身
    的重叠跨度 N（或 2N），保证每个块内部的自相关结构被完整保留在重采样里，
    而不是像普通 bootstrap 那样假设逐个观测独立（重叠 N 日窗口下这个假设不成立，
    会低估标准误、让置信区间虚窄）。
    """
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    block_len = max(1, min(block_len, n))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    starts_pool = np.arange(0, n - block_len + 1) if n > block_len else np.array([0])
    means = np.empty(reps)
    for r in range(reps):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        sample = np.concatenate([values[s : s + block_len] for s in starts])[:n]
        means[r] = np.nanmean(sample)
    lo, hi = np.nanquantile(means, ci)
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# 状态 -> 收敛概率对照表
# --------------------------------------------------------------------------- #

def estimate_convergence_table(
    events: pd.DataFrame,
    N: int,
    min_dte: Optional[int] = None,
    exclude_dividend_season: bool = False,
    cfg: config_mod.ReviewConfig = config_mod.DEFAULT_REVIEW_CONFIG,
) -> pd.DataFrame:
    """从收敛事件长表估计"z 分桶 -> 收敛概率 + 期望幅度"对照表。

    Args:
        events: build_convergence_events 的输出（可以是任意子集，例如
            neutral_hedge_timing.py 传入的某个 walk-forward 训练窗口切片）。
        N: 只用这个未来窗口对应的事件行。
        min_dte: 到期效应分层下限（剩余交易日 > min_dte 才纳入）；None 表示
            不过滤（对应"全样本对照"）。
        exclude_dividend_season: True 时剔除 touches_dividend_season 的事件
            （对应"剔除 5-7 月"的分红污染对照）。

    Returns:
        每个 z_bucket 一行：n_obs, p_converge, ci_low, ci_high,
        median_reduction_frac, q25_reduction_frac, q75_reduction_frac,
        median_days_to_converge。删失（converged 为 NaN）的行已被剔除，
        不计入分母。
    """
    sub = events[events["N"] == N].dropna(subset=["converged"]).copy()
    if min_dte is not None:
        sub = sub[sub["days_to_expiry"] > min_dte]
    if exclude_dividend_season:
        sub = sub[~sub["touches_dividend_season"]]
    sub = sub.sort_values("date")

    rows = []
    for bucket, grp in sub.groupby("z_bucket", observed=True):
        if grp.empty:
            continue
        converged_arr = grp["converged"].astype(float).to_numpy()
        lo, hi = block_bootstrap_ci(
            converged_arr, block_len=N, reps=cfg.convergence_bootstrap_reps,
            ci=cfg.convergence_bootstrap_ci, seed=cfg.convergence_bootstrap_seed,
        )
        rows.append(
            {
                "z_bucket": bucket,
                "n_obs": int(len(grp)),
                "p_converge": float(converged_arr.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "median_reduction_frac": float(grp["reduction_frac"].median()),
                "q25_reduction_frac": float(grp["reduction_frac"].quantile(0.25)),
                "q75_reduction_frac": float(grp["reduction_frac"].quantile(0.75)),
                "median_days_to_converge": float(grp["days_to_converge"].median()),
            }
        )
    return pd.DataFrame(rows)


def bucket_lookup_table(table: pd.DataFrame) -> dict[str, float]:
    """把 estimate_convergence_table 的结果压缩成 {z_bucket: p_converge} 字典，
    供 neutral_hedge_timing.py 做因果的"当天 bucket -> 概率"查表映射。"""
    if table.empty:
        return {}
    return dict(zip(table["z_bucket"].astype(str), table["p_converge"]))


# --------------------------------------------------------------------------- #
# 进阶（可选，未实现）
# --------------------------------------------------------------------------- #

def kaplan_meier_time_to_converge(*args, **kwargs):
    """TODO 升级钩子：用 Kaplan-Meier 生存分析估计"多久收敛"的完整分布
    （含删失样本），替代当前只报中位数天数的做法。未实现。"""
    raise NotImplementedError("升级钩子，尚未实现：Kaplan-Meier 生存分析")


def logit_convergence_model(*args, **kwargs):
    """TODO 升级钩子：logit 回归控制品种、剩余期限等协变量，估计收敛概率的
    连续函数形式，替代当前的离散分桶经验频率。未实现。"""
    raise NotImplementedError("升级钩子，尚未实现：logit 收敛概率回归")


# --------------------------------------------------------------------------- #
# 单指数编排 + 落盘
# --------------------------------------------------------------------------- #

def _plot_heatmap(
    main_tables: dict[int, pd.DataFrame],
    code: str,
    name: str,
    out_path: Path,
    edges: tuple[float, ...] = config_mod.CONVERGENCE_Z_BUCKET_EDGES,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    present = {b for t in main_tables.values() for b in t["z_bucket"].astype(str)}
    all_buckets = [b for b in bucket_labels(edges) if b in present]
    ns = sorted(main_tables.keys())
    matrix = np.full((len(all_buckets), len(ns)), np.nan)
    for j, N in enumerate(ns):
        table = main_tables[N].set_index(main_tables[N]["z_bucket"].astype(str))
        for i, bucket in enumerate(all_buckets):
            if bucket in table.index:
                matrix[i, j] = table.loc[bucket, "p_converge"]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_yticks(range(len(all_buckets)))
    ax.set_yticklabels(all_buckets)
    ax.set_title(f"{code} {name} 状态->收敛概率")
    for i in range(len(all_buckets)):
        for j in range(len(ns)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="p_converge")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_for_panel(
    factor_panel: pd.DataFrame,
    cfg: config_mod.ReviewConfig = config_mod.DEFAULT_REVIEW_CONFIG,
) -> pd.DataFrame:
    """跑完单个 (variant, index) 的事件构造 + 四种口径对照表，长表输出。

    四种口径的组合：
        min_dte 分层  in {cfg.convergence_min_dte(主), None(全样本对照)}
        dividend 过滤 in {False(默认，含分红季), True(剔除分红季对照)}
    每个 N 都跑全部组合，标注在 sample_filter / dividend_filter 两列，
    方便使用者按需筛选，而不是把四种口径拆成四个文件。
    """
    state_panel = attach_state(factor_panel, cfg)
    events = build_convergence_events(
        state_panel, n_horizons=cfg.convergence_n_horizons,
        std_threshold=cfg.convergence_std_threshold, decay_threshold=cfg.convergence_decay_threshold,
    )
    rows = []
    for N in cfg.convergence_n_horizons:
        for min_dte, sample_label in ((cfg.convergence_min_dte, "main_dte_filtered"), (None, "all_sample")):
            for exclude_div, div_label in ((False, "with_dividend_season"), (True, "ex_dividend_season")):
                table = estimate_convergence_table(events, N, min_dte, exclude_div, cfg)
                if table.empty:
                    continue
                table.insert(0, "N", N)
                table["sample_filter"] = sample_label
                table["dividend_filter"] = div_label
                rows.append(table)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined.attrs["events"] = events
    return combined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = config_mod.DEFAULT_REVIEW_CONFIG
    items = review_common_mod.discover_variant_indices()
    if not items:
        logger.warning("output/ 下没有任何 factor_panel.csv，先跑 run.py 再执行本模块。")
        return

    for item in items:
        panel = review_common_mod.load_factor_panel(item.factor_panel_path)
        combined = run_for_panel(panel, cfg)
        if combined.empty:
            logger.warning("[%s/%s] 历史长度不足以构造任何收敛事件，跳过。", item.variant, item.name)
            continue
        out_dir = review_common_mod.review_output_dir(
            review_common_mod.OUTPUT_ROOT, item.variant, cfg.output_subdir
        )
        combined.to_csv(out_dir / f"convergence_prob_{item.index_code}.csv", index=False)

        main_tables = {}
        for N in cfg.convergence_n_horizons:
            sub = combined[
                (combined["N"] == N)
                & (combined["sample_filter"] == "main_dte_filtered")
                & (combined["dividend_filter"] == "with_dividend_season")
            ]
            if not sub.empty:
                main_tables[N] = sub
        if main_tables:
            _plot_heatmap(
                main_tables, item.index_code, item.name,
                out_dir / f"convergence_prob_heatmap_{item.index_code}.png",
            )

        primary = combined[
            (combined["N"] == cfg.n_horizon)
            & (combined["sample_filter"] == "main_dte_filtered")
            & (combined["dividend_filter"] == "with_dividend_season")
        ]
        logger.info(
            "[%s/%s] 收敛概率（N=%d，主口径 dte>%d，含分红季）：\n%s",
            item.variant, item.name, cfg.n_horizon, cfg.convergence_min_dte,
            primary[["z_bucket", "n_obs", "p_converge", "ci_low", "ci_high"]].to_string(index=False)
            if not primary.empty else "(样本不足)",
        )


if __name__ == "__main__":
    main()
