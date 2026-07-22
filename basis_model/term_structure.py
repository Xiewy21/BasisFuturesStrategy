"""term_structure.py —— 期限结构向量化（基差复盘阶段一）。

方法论性质：**统计建模**（常数期限插值是确定性的工程变换，PCA 与 walk-forward
回归对比是统计建模）。

背景：factor_panel 每日只给出当月/次月/季月/下季月四个"标签口径"的年化基差，
但同一个标签在换月前后对应的实际剩余到期日会跳变（比如"当月"合约到期日从
还剩 3 天骤降到新合约的还剩 25 天左右），直接把 f1_ann_basis 按标签口径拼成
时间序列会带有这种换月锯齿，掩盖真实的期限结构变化。本模块先做"常数期限"
插值，把每天的期限结构曲线统一采样到 [30,60,90,180] 个自然日的固定网格上，
消除锯齿后再做 PCA/回归，这是把 f3（单一"季月-下季月"斜率因子）升级为
"整条曲线的主成分"之前必须先做的预处理。

流水线（对每个 (basis_variant, index_code) 独立进行）：
    1. build_term_vector      : 标签口径 4 维向量 V(t)，仅做透视，不插值。
    2. build_constant_maturity: 按 calendar_days_to_expiry 插值到常数期限网格，
       网格端点外不外推，标记 NaN（因此早期历史/换月附近某些网格点可能缺失）。
    3. fit_term_structure_pca : 对常数期限曲线做 PCA（只用完整行），产出前三个
       主成分载荷 + 方差解释比例 + 每日得分（PC_level/PC_slope/PC_curv）。
    4. compare_pca_vs_f3      : 把 PC_slope/PC_curv 得分广播回 factor_panel，
       与现有 f1+f3 模型并列跑 walk-forward 回归，比较系数显著性/R²/稳定性。

【防前视纪律】
    常数期限插值、PCA 载荷估计都只用"当天已发生"的截面数据（同一天 4 个合约
    的已实现价格），不跨日借用未来信息；PCA 载荷本身是用全部历史样本一次性
    估计的横截面降维（不是预测模型的参数），只在步骤 4 的 walk-forward 回归里
    才会因为"载荷用了全样本"而需要提醒读者：PC_slope/PC_curv 的【载荷】不是
    walk-forward 估计的，是这一步唯一没有做因果滚动的地方；真正做 walk-forward、
    因果滚动的是步骤 4 的回归系数本身（跟现有 regression.rolling_ols 一样，
    只用截止日之前的窗口拟合）。如果要在实盘每日更新 PC_slope/PC_curv，需要把
    步骤 3 的 PCA 也改成滚动重估（本模块暂未做，留待后续版本）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from basis_model import config as config_mod
from basis_model import regression as regression_mod
from basis_model import review_common as review_common_mod

logger = logging.getLogger("basis_model.term_structure")

MONTH_TAGS: tuple[str, ...] = ("当月", "次月", "季月", "下季月")


# --------------------------------------------------------------------------- #
# 步骤 1：标签口径向量
# --------------------------------------------------------------------------- #

def build_term_vector(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """按 month_tag 把每日四个合约的年化基差率拼成宽表 V(t)。

    Returns:
        DataFrame，index=date，columns=当月/次月/季月/下季月（缺失标签为 NaN）。
    """
    pivot = factor_panel.pivot_table(
        index="date", columns="month_tag", values="f1_ann_basis", aggfunc="first"
    )
    for tag in MONTH_TAGS:
        if tag not in pivot.columns:
            pivot[tag] = np.nan
    return pivot[list(MONTH_TAGS)].sort_index()


# --------------------------------------------------------------------------- #
# 步骤 2：常数期限插值
# --------------------------------------------------------------------------- #

def _interp_row(x: np.ndarray, y: np.ndarray, grid: np.ndarray, method: str) -> np.ndarray:
    """单日插值：只用当天有效的 (x, y) 点，网格端点外一律 NaN，不外推。"""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    out = np.full(grid.shape, np.nan)
    if len(x) < 2:
        return out
    order = np.argsort(x)
    x, y = x[order], y[order]
    # 同一剩余到期日出现重复合约时取均值，否则插值函数对重复 x 未定义。
    if len(np.unique(x)) < len(x):
        x, uniq_idx = np.unique(x, return_inverse=True)
        y = np.bincount(uniq_idx, weights=y) / np.bincount(uniq_idx)
    in_range = (grid >= x.min()) & (grid <= x.max())
    if not in_range.any():
        return out
    if method == "linear":
        out[in_range] = np.interp(grid[in_range], x, y)
    elif method == "cubic":
        from scipy.interpolate import CubicSpline

        if len(x) < 3:
            out[in_range] = np.interp(grid[in_range], x, y)
        else:
            spline = CubicSpline(x, y, extrapolate=False)
            out[in_range] = spline(grid[in_range])
    else:
        raise ValueError(f"未知插值方法: {method!r}，只支持 'linear'/'cubic'")
    return out


def build_constant_maturity(
    factor_panel: pd.DataFrame,
    grid_days: tuple[int, ...] = config_mod.TERM_GRID_DAYS,
    method: str = config_mod.TERM_INTERP_METHOD,
) -> pd.DataFrame:
    """把每日四个 (calendar_days_to_expiry, f1_ann_basis) 点插值到常数期限网格。

    x 轴用 calendar_days_to_expiry（自然日）而非 days_to_expiry（交易日），
    因为网格本身以自然日定义（30/60/90/180 个自然日），单位必须一致。
    网格端点外（当天最大剩余自然日 < 网格值，或最小剩余自然日 > 网格值）
    一律标记 NaN，不外推。

    Returns:
        DataFrame，index=date，columns=grid_days 对应的常数期限列（如 cm_30）。
    """
    grid = np.array(grid_days, dtype=float)
    cols = [f"cm_{d}" for d in grid_days]
    rows = []
    dates = []
    for d, grp in factor_panel.groupby("date"):
        x = grp["calendar_days_to_expiry"].to_numpy(dtype=float)
        y = grp["f1_ann_basis"].to_numpy(dtype=float)
        rows.append(_interp_row(x, y, grid, method))
        dates.append(d)
    out = pd.DataFrame(rows, index=pd.Index(dates, name="date"), columns=cols).sort_index()
    return out


# --------------------------------------------------------------------------- #
# 步骤 3：PCA
# --------------------------------------------------------------------------- #

def _classify_component(loading: np.ndarray) -> str:
    """按载荷形状粗分 level/slope/curvature：符号变化次数 0/1/>=2 次。"""
    signs = np.sign(loading)
    signs = signs[signs != 0]
    if len(signs) == 0:
        return "level"
    changes = int((np.diff(signs) != 0).sum())
    if changes == 0:
        return "level"
    if changes == 1:
        return "slope"
    return "curvature"


def fit_term_structure_pca(
    constant_maturity: pd.DataFrame,
    n_components: int = config_mod.TERM_PCA_N_COMPONENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对常数期限曲线做 PCA，只用四个网格点全部非 NaN 的"完整日"拟合。

    Returns:
        (scores, loadings, explained_variance)
            scores: index=date（仅完整日），columns=PC1/PC2/PC3（前 n_components 个）
            loadings: index=grid 列名，columns=PC1/PC2/PC3，另有 component_type 列
            explained_variance: 每个 PC 的方差解释比例
    """
    complete = constant_maturity.dropna(how="any")
    if len(complete) < n_components + 1:
        empty_scores = pd.DataFrame(columns=[f"PC{i+1}" for i in range(n_components)])
        empty_loadings = pd.DataFrame(
            columns=[f"PC{i+1}" for i in range(n_components)] + ["component_type"]
        )
        empty_var = pd.DataFrame(columns=["component", "explained_variance_ratio", "component_type"])
        return empty_scores, empty_loadings, empty_var

    mean = complete.mean(axis=0)
    centered = (complete - mean).to_numpy()
    # SVD 而非协方差矩阵特征分解：数值更稳定，等价结果。
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(n_components, Vt.shape[0])
    loadings_arr = Vt[:n_components].T  # (n_grid, n_components)
    scores_arr = centered @ loadings_arr

    pc_names = [f"PC{i+1}" for i in range(n_components)]
    component_types = [_classify_component(loadings_arr[:, i]) for i in range(n_components)]
    loadings = pd.DataFrame(loadings_arr, index=complete.columns, columns=pc_names)
    # component_type 是"每个 PC 一个值"的元数据，不是随网格行变化的数据，
    # 因此追加成一行（而非一列），保持 loadings 矩阵形状 = (n_grid, n_pc)。
    loadings.loc["component_type"] = component_types

    total_var = (S**2).sum()
    explained = (S[:n_components] ** 2) / total_var
    explained_variance = pd.DataFrame(
        {
            "component": pc_names,
            "explained_variance_ratio": explained,
            "component_type": component_types,
        }
    )

    scores = pd.DataFrame(scores_arr, index=complete.index, columns=pc_names)
    scores.index.name = "date"
    return scores, loadings, explained_variance


#: _classify_component 输出（level/slope/curvature）到下游回归自变量列名前缀的映射；
#: "curvature" 缩写成 "curv"，与用户口径 "b*PC_slope + c*PC_curv" 的列名保持一致。
_COMPONENT_TYPE_TO_PREFIX: dict[str, str] = {
    "level": "PC_level",
    "slope": "PC_slope",
    "curvature": "PC_curv",
}


def _rename_scores_by_type(scores: pd.DataFrame, explained_variance: pd.DataFrame) -> pd.DataFrame:
    """把 PC1/PC2/PC3 按载荷形状改名为 PC_level/PC_slope/PC_curv，方便下游引用。

    同一形状出现多次时（少见），后出现的加数字后缀，保证列名不冲突。
    """
    type_map = dict(zip(explained_variance["component"], explained_variance["component_type"]))
    seen: dict[str, int] = {}
    rename = {}
    for pc in scores.columns:
        base = _COMPONENT_TYPE_TO_PREFIX.get(type_map.get(pc, "other"), "PC_other")
        seen[base] = seen.get(base, 0) + 1
        rename[pc] = base if seen[base] == 1 else f"{base}{seen[base]}"
    return scores.rename(columns=rename)


# --------------------------------------------------------------------------- #
# 步骤 4：PCA 斜率/曲率替代 f3 的 walk-forward 回归对比
# --------------------------------------------------------------------------- #

def compare_pca_vs_f3(
    factor_panel: pd.DataFrame,
    scores_named: pd.DataFrame,
    cfg: config_mod.ReviewConfig = config_mod.DEFAULT_REVIEW_CONFIG,
) -> pd.DataFrame:
    """并列跑两组 walk-forward 回归：baseline(f1+f3) vs pca(f1+PC_slope+PC_curv)。

    PC 得分按 date 广播到 factor_panel 每一行（与 f3_term_structure 的广播方式
    一致：同一天所有合约共享同一条期限结构曲线，因此共享同一组 PC 得分）。
    只在两组模型都能拿到非 NaN 自变量的日期上比较，避免样本不对齐导致的
    虚假差异。

    Returns:
        长表：列 = date, model('baseline_f3'/'pca_slope_curv'), n_obs, r_squared,
        const, 各自变量系数, tstat_*, pvalue_*, D_month_*。
    """
    merged = factor_panel.merge(
        scores_named.reset_index(), on="date", how="left", validate="many_to_one"
    )

    has_slope = "PC_slope" in scores_named.columns
    has_curv = "PC_curv" in scores_named.columns
    pca_cols = ["f1_ann_basis"] + (["PC_slope"] if has_slope else []) + (
        ["PC_curv"] if has_curv else []
    )

    baseline_coefs = regression_mod.rolling_ols(
        merged,
        window=cfg.rolling_window,
        min_obs=cfg.min_obs,
        step=cfg.refit_every,
        feature_cols=list(regression_mod.DEFAULT_FEATURE_COLS),
        include_month_dummies=True,
        hac_lags=cfg.n_horizon,
    ).reset_index()
    baseline_coefs.insert(1, "model", "baseline_f3")

    if len(pca_cols) < 2:
        pca_coefs = pd.DataFrame(columns=baseline_coefs.columns)
    else:
        pca_coefs = regression_mod.rolling_ols(
            merged,
            window=cfg.rolling_window,
            min_obs=cfg.min_obs,
            step=cfg.refit_every,
            feature_cols=pca_cols,
            include_month_dummies=True,
            hac_lags=cfg.n_horizon,
        ).reset_index()
        pca_coefs.insert(1, "model", "pca_slope_curv")

    return pd.concat([baseline_coefs, pca_coefs], ignore_index=True, sort=False)


def summarize_stability(comparison: pd.DataFrame) -> pd.DataFrame:
    """从 compare_pca_vs_f3 的输出里，为每个模型的"斜率类"系数算稳定性摘要。

    稳定性 = walk-forward 系数序列的标准差（越小越稳），显著份额 = |tstat|>=1.96
    的窗口占比。这是"PCA 斜率因子是否比 f3 更稳"这个待验证假设的直接量化依据。
    """
    rows = []
    slope_col_by_model = {"baseline_f3": "f3_term_structure", "pca_slope_curv": "PC_slope"}
    for model, slope_col in slope_col_by_model.items():
        sub = comparison[comparison["model"] == model]
        if sub.empty or slope_col not in sub.columns:
            rows.append(
                {
                    "model": model,
                    "slope_factor": slope_col,
                    "n_windows": 0,
                    "coef_mean": np.nan,
                    "coef_std": np.nan,
                    "mean_r_squared": np.nan,
                    "significant_share": np.nan,
                }
            )
            continue
        tstat_col = f"tstat_{slope_col}"
        significant_share = (
            float((sub[tstat_col].abs() >= 1.96).mean()) if tstat_col in sub.columns else np.nan
        )
        rows.append(
            {
                "model": model,
                "slope_factor": slope_col,
                "n_windows": int(len(sub)),
                "coef_mean": float(sub[slope_col].mean()),
                "coef_std": float(sub[slope_col].std(ddof=1)) if len(sub) > 1 else 0.0,
                "mean_r_squared": float(sub["r_squared"].mean()),
                "significant_share": significant_share,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 单指数编排 + 落盘
# --------------------------------------------------------------------------- #

def run_for_panel(
    factor_panel: pd.DataFrame,
    cfg: config_mod.ReviewConfig = config_mod.DEFAULT_REVIEW_CONFIG,
) -> dict[str, pd.DataFrame]:
    """跑完单个 (variant, index) 的四个步骤，返回全部中间结果供落盘/单测使用。"""
    term_vector = build_term_vector(factor_panel)
    constant_maturity = build_constant_maturity(
        factor_panel, grid_days=cfg.term_grid_days, method=cfg.term_interp_method
    )
    scores, loadings, explained_variance = fit_term_structure_pca(
        constant_maturity, n_components=cfg.term_pca_n_components
    )
    scores_named = (
        _rename_scores_by_type(scores, explained_variance) if not scores.empty else scores
    )
    comparison = (
        compare_pca_vs_f3(factor_panel, scores_named, cfg)
        if not scores_named.empty
        else pd.DataFrame()
    )
    stability = summarize_stability(comparison) if not comparison.empty else pd.DataFrame()
    return {
        "term_vector": term_vector,
        "constant_maturity": constant_maturity,
        "pca_scores": scores_named,
        "pca_loadings": loadings,
        "pca_explained_variance": explained_variance,
        "regression_comparison": comparison,
        "stability_summary": stability,
    }


def _write_outputs(
    out_dir: Path, code: str, results: dict[str, pd.DataFrame]
) -> None:
    results["term_vector"].to_csv(out_dir / f"term_vector_{code}.csv")
    results["pca_loadings"].to_csv(out_dir / f"pca_loadings_{code}.csv")
    results["regression_comparison"].to_csv(
        out_dir / f"regression_pca_vs_f3_{code}.csv", index=False
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = config_mod.DEFAULT_REVIEW_CONFIG
    items = review_common_mod.discover_variant_indices()
    if not items:
        logger.warning("output/ 下没有任何 factor_panel.csv，先跑 run.py 再执行本模块。")
        return

    explained_variance_rows = []
    verdict_rows = []
    for item in items:
        panel = review_common_mod.load_factor_panel(item.factor_panel_path)
        results = run_for_panel(panel, cfg)
        out_dir = review_common_mod.review_output_dir(
            review_common_mod.OUTPUT_ROOT, item.variant, cfg.output_subdir
        )
        _write_outputs(out_dir, item.index_code, results)

        ev = results["pca_explained_variance"].copy()
        if not ev.empty:
            ev.insert(0, "index_code", item.index_code)
            ev.insert(0, "basis_variant", item.variant)
            explained_variance_rows.append(ev)

        stability = results["stability_summary"]
        if not stability.empty:
            baseline = stability[stability["model"] == "baseline_f3"].iloc[0]
            pca_row = stability[stability["model"] == "pca_slope_curv"].iloc[0]
            more_stable = (
                pca_row["coef_std"] < baseline["coef_std"]
                if pd.notna(pca_row["coef_std"]) and pd.notna(baseline["coef_std"])
                else None
            )
            verdict_rows.append(
                {
                    "basis_variant": item.variant,
                    "index_code": item.index_code,
                    "name": item.name,
                    "f3_coef_std": baseline["coef_std"],
                    "f3_significant_share": baseline["significant_share"],
                    "pc_slope_coef_std": pca_row["coef_std"],
                    "pc_slope_significant_share": pca_row["significant_share"],
                    "pca_slope_more_stable_than_f3": more_stable,
                }
            )
        logger.info(
            "[%s/%s] 期限结构：完整常数期限日 %d/%d，PCA 前 %d 主成分方差解释 %s",
            item.variant,
            item.name,
            results["pca_scores"].shape[0] if not results["pca_scores"].empty else 0,
            len(results["constant_maturity"]),
            cfg.term_pca_n_components,
            ev["explained_variance_ratio"].round(3).tolist() if not ev.empty else [],
        )

    if explained_variance_rows:
        combined_ev = pd.concat(explained_variance_rows, ignore_index=True)
        for variant, grp in combined_ev.groupby("basis_variant"):
            out_dir = review_common_mod.review_output_dir(
                review_common_mod.OUTPUT_ROOT, variant, cfg.output_subdir
            )
            grp.drop(columns=["basis_variant"]).to_csv(
                out_dir / "pca_explained_variance.csv", index=False
            )

    if verdict_rows:
        verdict = pd.DataFrame(verdict_rows)
        logger.info(
            "待验证假设「PCA 斜率因子是否比 f3 更稳」结论（系数标准差越小越稳）：\n%s",
            verdict.to_string(index=False),
        )
    else:
        logger.warning("没有任何指数产出足够的完整常数期限样本用于 PCA 回归对比。")


if __name__ == "__main__":
    main()
