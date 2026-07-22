"""basis_center.py —— 基差中枢与异常识别（基差复盘阶段二）。

方法论性质：**统计建模**（滚动中位数/分位数、z-score 都是描述性统计，不含
可训练参数，但选取"用中位数而非均值""因果滚动"本身是建模选择）。

为什么用中位数而不是均值：基差（尤其是年化基差率）在分红季、临近交割、
极端行情下会出现肥尾/跳变，均值会被这些极值样本明显拖偏，中位数对这类
污染更稳健。长期中枢单独给一份全样本/分段中位数供参考对照，但【不】用于
z-score（z-score 必须用因果滚动中枢，否则相当于用未来数据定义"正常水平"）。

核心口径：
    中枢分两套坐标系分别算，long-format 拼在同一张表里（period_type 列区分）：
        - period_type='label'：按 (index_code, month_tag) 分组 —— 与
          term_structure.py 的"标签口径"一致，把同一 month_tag 下轮换的合约
          近似当成一条连续序列。
        - period_type='constant_maturity'：按 (index_code, cm_<grid>) 分组，
          cm_<grid> 取自 term_structure.build_constant_maturity 的常数期限插值
          曲线（30/60/90/180 自然日）。换月时同一个 month_tag 标签背后的合约
          剩余到期日会跳变（比如"当月"从还剩 3 天骤降到新合约的 25 天），
          直接在标签口径上滚动会带这种锯齿；常数期限口径先插值到固定期限，
          再滚动中枢，跨日可比、没有换月跳变，是对标签口径的补充对照，不是替代。
    两套口径都用同一套函数（causal_rolling_stats / causal_rolling_percentile /
    long_term_center / compute_z_score / flag_anomalies），只是分组键从
    month_tag 换成 period（通用列名），逻辑完全一致。

    当期中枢 = 250 日滚动中位数（另跑 500 日对照），滚动通道 = 25/50/75 分位，
    另外输出滚动分位 p(t) ∈ [0,1]（今天的值在过去历史里的分位排名）。
    z(t) = (f1(t) - center_250(t)) / std_250(t)，中枢/std/p(t) 都只用
    date < t 的历史（p(t) 例外：用 [t-window, t] 共 window+1 个点算排名，
    但排名比较的对象仍是"t 之前的 window 个点"，t 自己只是被比较的对象，
    不参与定义比较基准，因此同样只用 ≤t 的数据、不泄漏未来）。

异常识别两套口径，独立标注、不合并成一个布尔值：
    - 统计口径：当天值跌破/突破滚动 5%/95% 分位阈值，或 |z|>2。
    - 持续性口径：统计口径连续触发 >= m(=3) 个交易日（因果定义：走到第 m 天
      才承认"进入持续异常"，不回溯标注前面 m-1 天，避免把"事后才知道会持续"
      的信息提前标到过去）。
    另外单独标注 is_dividend_season（5-7月）：raw 口径基差未剔分红，这段时间
    的异常可能是分红导致的假异常，必须能被使用者过滤掉，而不是被模型自动剔除
    （剔除与否留给下游判断，本模块只负责如实标注）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from basis_model import config as config_mod
from basis_model import dividend as dividend_mod
from basis_model import review_common as review_common_mod
from basis_model import term_structure as term_structure_mod

logger = logging.getLogger("basis_model.basis_center")


# --------------------------------------------------------------------------- #
# 因果滚动中枢 / 通道 / z-score / 滚动分位排名
# --------------------------------------------------------------------------- #

def causal_rolling_stats(
    f1: pd.Series,
    index_code: pd.Series,
    period: pd.Series,
    date: pd.Series,
    window: int,
    min_periods: int,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> pd.DataFrame:
    """按 (index_code, period) 分组、date 排序，产出 t 日因果滚动统计量。

    period 可以是标签口径的 month_tag（当月/次月/季月/下季月），也可以是
    常数期限口径的 cm_30/cm_60/cm_90/cm_180 —— 两套坐标系复用同一份实现，
    调用方（build_basis_center_table）分别传入即可。

    先 shift(1) 再滚动：t 日的中枢/std/分位数只用 date < t 的历史观测，
    与 strategy.causal_basis_thresholds 同一纪律，是本模块"因果滚动"的唯一
    实现入口，其余函数一律调用本函数，不重复实现 shift+rolling 逻辑。

    Returns:
        DataFrame，与输入等长（reindex 回原始 index），列：
        center（滚动中位数）、std（滚动标准差）、q<pct*100>（各分位数）。
    """
    orig_index = f1.index
    frame = pd.DataFrame(
        {
            "f1": f1.to_numpy(),
            "index_code": index_code.to_numpy(),
            "period": period.to_numpy(),
            "date": pd.to_datetime(date).to_numpy(),
        },
        index=orig_index,
    ).sort_values(["index_code", "period", "date"])

    history = frame.groupby(["index_code", "period"])["f1"].shift(1)
    grouped_history = history.groupby([frame["index_code"], frame["period"]])

    out = pd.DataFrame(index=frame.index)
    out["center"] = grouped_history.transform(
        lambda s: s.rolling(window, min_periods=min_periods).median()
    )
    out["std"] = grouped_history.transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )
    for q in quantiles:
        out[f"q{int(round(q * 100)):02d}"] = grouped_history.transform(
            lambda s, q=q: s.rolling(window, min_periods=min_periods).quantile(q)
        )
    return out.reindex(orig_index)


def causal_rolling_percentile(
    f1: pd.Series,
    index_code: pd.Series,
    period: pd.Series,
    date: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """p(t) = f1(t) 在 [t-window, t-1] 这段严格早于 t 的历史里的分位排名 ∈[0,1]。

    实现上直接在原始（未 shift）序列上取 window+1 长度的滚动窗口，窗口内最后一个
    元素天然就是 t 当天的观测，其余 window 个元素才是比较基准；因此不需要像
    causal_rolling_stats 那样显式 shift(1) —— 当天的值只作为"被比较的对象"，
    从不参与"比较基准"本身，同样满足因果口径。
    """
    orig_index = f1.index
    frame = pd.DataFrame(
        {
            "f1": f1.to_numpy(),
            "index_code": index_code.to_numpy(),
            "period": period.to_numpy(),
            "date": pd.to_datetime(date).to_numpy(),
        },
        index=orig_index,
    ).sort_values(["index_code", "period", "date"])

    def _pct_rank(arr: np.ndarray) -> float:
        history, current = arr[:-1], arr[-1]
        if np.isnan(current):
            return np.nan
        return float(np.mean(history < current))

    grouped = frame.groupby(["index_code", "period"])["f1"]
    p = grouped.transform(
        lambda s: s.rolling(window + 1, min_periods=min_periods + 1).apply(_pct_rank, raw=True)
    )
    return p.reindex(orig_index)


def long_term_center(
    f1: pd.Series, index_code: pd.Series, period: pd.Series, date: pd.Series
) -> pd.DataFrame:
    """长期中枢：全样本中位数 + 按年分段中位数，仅供参考展示，不参与 z-score。

    与 causal_rolling_stats 的关键区别：这里故意用了全部历史（含未来于当前
    观测点的样本），因此严禁用它计算任何会被当作实时信号使用的量，只能作为
    报表里的"历史整体水平"对照列。
    """
    frame = pd.DataFrame(
        {
            "f1": f1.to_numpy(),
            "index_code": index_code.to_numpy(),
            "period": period.to_numpy(),
            "year": pd.to_datetime(date).dt.year.to_numpy(),
        },
        index=f1.index,
    )
    full_sample = frame.groupby(["index_code", "period"])["f1"].transform("median")
    by_year = frame.groupby(["index_code", "period", "year"])["f1"].transform("median")
    return pd.DataFrame(
        {"center_long_full_sample": full_sample, "center_long_annual": by_year},
        index=f1.index,
    )


def compute_z_score(f1: pd.Series, center: pd.Series, std: pd.Series) -> pd.Series:
    """z(t) = (f1(t) - center(t)) / std(t)；std<=0 或 NaN 时返回 NaN。"""
    std_safe = std.where(std > 0, other=np.nan)
    return (f1 - center) / std_safe


# --------------------------------------------------------------------------- #
# 异常识别
# --------------------------------------------------------------------------- #

def flag_anomalies(
    f1: pd.Series,
    index_code: pd.Series,
    period: pd.Series,
    date: pd.Series,
    rolling_stats: pd.DataFrame,
    z: pd.Series,
    tail_pct: float = config_mod.ANOMALY_TAIL_PCT,
    z_threshold: float = config_mod.ANOMALY_Z_THRESHOLD,
    persistence_m: int = config_mod.ANOMALY_PERSISTENCE_M,
) -> pd.DataFrame:
    """两套异常口径 + 分红季标注，独立成列，不合并成单一布尔值。

    Args:
        rolling_stats: causal_rolling_stats 的输出，需要 q<lower>/q<upper> 列
            覆盖 tail_pct 对应的分位（默认 5%/95%，取 causal_rolling_stats
            额外算好的两端分位；本函数不重新滚动计算，直接复用传入结果，
            避免同一份滚动窗口算两遍）。

    Returns:
        DataFrame（与输入等长）：is_extreme_quantile, is_extreme_z,
        is_statistical_anomaly, run_length, is_persistent_anomaly,
        anomaly_type（"persistent"/"statistical"/"none"，供 anomaly_events 直接
        标注类型，而不需要使用者自己拼 is_extreme_*/is_persistent_anomaly 三列）,
        is_dividend_season。
    """
    lower_col = f"q{int(round(tail_pct * 100)):02d}"
    upper_col = f"q{int(round((1 - tail_pct) * 100)):02d}"
    if lower_col not in rolling_stats.columns or upper_col not in rolling_stats.columns:
        raise ValueError(
            f"rolling_stats 缺少 {lower_col}/{upper_col} 分位列，请在 "
            "causal_rolling_stats 的 quantiles 参数里包含 tail_pct 对应分位。"
        )

    is_extreme_quantile = (f1 < rolling_stats[lower_col]) | (f1 > rolling_stats[upper_col])
    is_extreme_z = z.abs() > z_threshold
    is_statistical_anomaly = (is_extreme_quantile | is_extreme_z).fillna(False)

    frame = pd.DataFrame(
        {
            "flag": is_statistical_anomaly.to_numpy(),
            "index_code": index_code.to_numpy(),
            "period": period.to_numpy(),
            "date": pd.to_datetime(date).to_numpy(),
        },
        index=f1.index,
    ).sort_values(["index_code", "period", "date"])

    grouped = frame.groupby(["index_code", "period"])["flag"]
    # 连续 True 的游程长度：每遇到一个 False 就重开一段计数（经典 groupby+cumsum 技巧）。
    run_reset = (~frame["flag"]).groupby([frame["index_code"], frame["period"]]).cumsum()
    run_length = frame.groupby(
        [frame["index_code"], frame["period"], run_reset]
    )["flag"].cumsum()
    frame["run_length"] = np.where(frame["flag"], run_length, 0)
    frame["is_persistent_anomaly"] = frame["flag"] & (frame["run_length"] >= persistence_m)

    result = frame[["run_length", "is_persistent_anomaly"]].reindex(f1.index)
    result.insert(0, "is_statistical_anomaly", is_statistical_anomaly)
    result.insert(0, "is_extreme_z", is_extreme_z.fillna(False))
    result.insert(0, "is_extreme_quantile", is_extreme_quantile.fillna(False))
    result["anomaly_type"] = np.select(
        [result["is_persistent_anomaly"], result["is_statistical_anomaly"]],
        ["persistent", "statistical"],
        default="none",
    )
    result["is_dividend_season"] = pd.to_datetime(date).map(dividend_mod.is_dividend_season)
    return result


# --------------------------------------------------------------------------- #
# 单指数编排 + 落盘
# --------------------------------------------------------------------------- #

def _compute_center_and_anomalies(
    f1: pd.Series,
    index_code: pd.Series,
    period: pd.Series,
    date: pd.Series,
    cfg: config_mod.ReviewConfig,
) -> pd.DataFrame:
    """中枢(250/500) + 长期中枢 + 滚动分位 p + z-score + 异常标注，通用于标签口径
    和常数期限口径 —— 两者只是分组用的 period 定义不同，计算逻辑完全一致。
    """
    quantiles = tuple(sorted({*cfg.basis_center_quantiles, cfg.anomaly_tail_pct, 1 - cfg.anomaly_tail_pct}))

    stats_250_raw = causal_rolling_stats(
        f1, index_code, period, date, window=cfg.basis_center_window,
        min_periods=cfg.basis_center_min_periods, quantiles=quantiles,
    )
    stats_250 = stats_250_raw.add_suffix("_250")
    stats_500 = causal_rolling_stats(
        f1, index_code, period, date, window=cfg.basis_center_window_alt,
        min_periods=cfg.basis_center_min_periods, quantiles=cfg.basis_center_quantiles,
    ).add_suffix("_500")
    p_250 = causal_rolling_percentile(
        f1, index_code, period, date, window=cfg.basis_center_window,
        min_periods=cfg.basis_center_min_periods,
    )
    p_500 = causal_rolling_percentile(
        f1, index_code, period, date, window=cfg.basis_center_window_alt,
        min_periods=cfg.basis_center_min_periods,
    )
    long_term = long_term_center(f1, index_code, period, date)
    z_250 = compute_z_score(f1, stats_250["center_250"], stats_250["std_250"])
    z_500 = compute_z_score(f1, stats_500["center_500"], stats_500["std_500"])

    anomalies = flag_anomalies(
        f1, index_code, period, date, stats_250_raw, z_250,
        tail_pct=cfg.anomaly_tail_pct, z_threshold=cfg.anomaly_z_threshold,
        persistence_m=cfg.anomaly_persistence_m,
    )

    out = pd.concat([stats_250, stats_500, long_term], axis=1)
    out["p_250"] = p_250
    out["p_500"] = p_500
    out["z_250"] = z_250
    out["z_500"] = z_500
    out = pd.concat([out, anomalies], axis=1)
    return out


def build_basis_center_table(
    factor_panel: pd.DataFrame,
    cfg: config_mod.ReviewConfig = config_mod.DEFAULT_REVIEW_CONFIG,
) -> pd.DataFrame:
    """跑完中枢 + z-score + 异常标注的完整长表，纵向拼接两套坐标系：

        period_type='label'：一行 = 一个 (date, contract) 观测，period=month_tag。
        period_type='constant_maturity'：一行 = 一个 (date, cm_<grid>) 观测
            （term_structure.build_constant_maturity 插值出的常数期限曲线），
            没有单一 contract，contract 列为 NaN。

    factor_panel 必须只含单一 index_code（main() 按 (variant, index) 逐个调用，
    与其余复盘模块的编排方式一致）。
    """
    index_codes = factor_panel["index_code"].unique()
    if len(index_codes) != 1:
        raise ValueError(f"build_basis_center_table 需要单一 index_code，实际: {index_codes}")
    index_code_value = index_codes[0]

    label_base = factor_panel[["date", "contract", "index_code", "month_tag", "f1_ann_basis"]].rename(
        columns={"month_tag": "period"}
    ).reset_index(drop=True)
    label_base.insert(4, "period_type", "label")
    label_stats = _compute_center_and_anomalies(
        factor_panel["f1_ann_basis"].reset_index(drop=True),
        factor_panel["index_code"].reset_index(drop=True),
        factor_panel["month_tag"].reset_index(drop=True),
        factor_panel["date"].reset_index(drop=True),
        cfg,
    )
    label_table = pd.concat([label_base, label_stats], axis=1)

    cm = term_structure_mod.build_constant_maturity(
        factor_panel, grid_days=cfg.term_grid_days, method=cfg.term_interp_method
    )
    cm_long = cm.reset_index().melt(id_vars="date", var_name="period", value_name="f1_ann_basis")
    cm_long = cm_long.sort_values(["period", "date"]).reset_index(drop=True)
    cm_long.insert(1, "contract", np.nan)
    cm_long.insert(2, "index_code", index_code_value)
    cm_long.insert(4, "period_type", "constant_maturity")
    cm_stats = _compute_center_and_anomalies(
        cm_long["f1_ann_basis"], cm_long["index_code"], cm_long["period"], cm_long["date"], cfg,
    )
    cm_table = pd.concat([cm_long, cm_stats], axis=1)

    return pd.concat([label_table, cm_table], ignore_index=True, sort=False)


def _plot_ten_year_chart(table: pd.DataFrame, code: str, name: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 图内中文标签需要一个内置 CJK 字形的字体，否则会被渲染成方框；
    # macOS 自带 PingFang SC，缺失时静默回退到默认字体（英文标题仍可读）。
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    month_tags = ["当月", "次月", "季月", "下季月"]
    label_table = table[table["period_type"] == "label"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, tag in zip(axes.flat, month_tags):
        sub = label_table[label_table["period"] == tag].sort_values("date")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub["date"], sub["f1_ann_basis"], color="#4C72B0", linewidth=0.8, label="年化基差率")
        ax.plot(sub["date"], sub["center_250"], color="#DD8452", linewidth=1.2, label="250日滚动中枢")
        ax.fill_between(
            sub["date"], sub["q25_250"], sub["q75_250"], color="#DD8452", alpha=0.15,
            label="25%-75%滚动通道",
        )
        ax.axhline(
            sub["center_long_full_sample"].iloc[0], color="gray", linestyle="--", linewidth=0.8,
            label="全样本中位数",
        )
        statistical_only = sub[sub["anomaly_type"] == "statistical"]
        persistent = sub[sub["anomaly_type"] == "persistent"]
        ax.scatter(
            statistical_only["date"], statistical_only["f1_ann_basis"], color="#E8A33D",
            s=10, zorder=3, label="统计异常",
        )
        ax.scatter(
            persistent["date"], persistent["f1_ann_basis"], color="#C44E52",
            s=14, zorder=4, label="持续异常",
        )
        ax.set_title(tag)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{code} {name} 年化基差率十年中枢")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = config_mod.DEFAULT_REVIEW_CONFIG
    items = review_common_mod.discover_variant_indices()
    if not items:
        logger.warning("output/ 下没有任何 factor_panel.csv，先跑 run.py 再执行本模块。")
        return

    for item in items:
        panel = review_common_mod.load_factor_panel(item.factor_panel_path)
        table = build_basis_center_table(panel, cfg)
        out_dir = review_common_mod.review_output_dir(
            review_common_mod.OUTPUT_ROOT, item.variant, cfg.output_subdir
        )
        table.to_csv(out_dir / f"basis_center_{item.index_code}.csv", index=False)

        anomalies_only = table[table["anomaly_type"] != "none"]
        anomalies_only.to_csv(out_dir / f"anomaly_events_{item.index_code}.csv", index=False)

        _plot_ten_year_chart(
            table, item.index_code, item.name, out_dir / f"basis_center_{item.index_code}.png"
        )

        label_table = table[table["period_type"] == "label"]
        n_persistent = int(label_table["is_persistent_anomaly"].sum())
        n_dividend_overlap = int(
            (label_table["is_persistent_anomaly"] & label_table["is_dividend_season"]).sum()
        )
        logger.info(
            "[%s/%s] 中枢：%d 行（标签口径），统计异常 %d 行，持续异常 %d 行（其中 %d 行落在"
            "分红季，占持续异常的 %.1f%%）",
            item.variant,
            item.name,
            len(label_table),
            int(label_table["is_statistical_anomaly"].sum()),
            n_persistent,
            n_dividend_overlap,
            (n_dividend_overlap / n_persistent * 100) if n_persistent else 0.0,
        )


if __name__ == "__main__":
    main()
