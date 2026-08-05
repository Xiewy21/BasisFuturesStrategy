"""chart_export.py —— （纯展示层）把期限结构 / 基差回归系数复盘结果导出成图表。

方法论性质：无（纯展示层，不产出任何新的模型结论）。期限结构面板重新调用
term_structure.build_constant_maturity / fit_term_structure_pca 对 factor_panel
现算（这两步本身是确定性变换/横截面 PCA，现算与读盘等价，避免额外维护一份
常数期限曲线的落盘文件）；基差系数面板直接读 run.py 已经写好的
regression_coefs.csv，不重新拟合。

输出固定落到 output/结果图表/{期限结构,基差系数}/，每个 basis_variant 一张图。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from basis_model import config as config_mod
from basis_model import review_common as review_common_mod
from basis_model import term_structure as term_structure_mod

logger = logging.getLogger("basis_model.chart_export")

CHART_ROOT: Path = review_common_mod.OUTPUT_ROOT / "结果图表"
TERM_STRUCTURE_CHART_DIR: Path = CHART_ROOT / "期限结构"
BASIS_COEF_CHART_DIR: Path = CHART_ROOT / "基差系数"

#: 四个中金所股指期货品种的固定配色（分类色，按品种固定映射，不随出现顺序改变）。
_INDEX_COLORS: dict[str, str] = {
    "000016": "#1b3358",  # IH / 上证50
    "000300": "#4a7ab5",  # IF / 沪深300
    "000905": "#c0392b",  # IC / 中证500
    "000852": "#2e8b57",  # IM / 中证1000
}
_INDEX_SHORT_NAMES: dict[str, str] = {
    "000016": "IH",
    "000300": "IF",
    "000905": "IC",
    "000852": "IM",
}
_FALLBACK_PALETTE: tuple[str, ...] = ("#1b3358", "#4a7ab5", "#c0392b", "#2e8b57", "#8e6c8a", "#c98a2e")


def _color_for(code: str, fallback_idx: int) -> str:
    return _INDEX_COLORS.get(code, _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)])


def _label_for(code: str, name: str) -> str:
    short = _INDEX_SHORT_NAMES.get(code)
    return f"{short} / {name}" if short else f"{code} {name}"


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 图内中文标签需要一个内置 CJK 字形的字体，否则会被渲染成方框；
    # macOS 自带 PingFang SC，缺失时静默回退到默认字体（英文标题仍可读）。
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# --------------------------------------------------------------------------- #
# 期限结构（output/结果图表/期限结构/）
# --------------------------------------------------------------------------- #

def _term_structure_panel_data(
    items: list[review_common_mod.VariantIndex],
    cfg: config_mod.ReviewConfig,
) -> tuple[dict[str, tuple[str, pd.Series]], dict[str, tuple[str, pd.DataFrame]], pd.DataFrame]:
    """对每个 index 现算常数期限曲线均值 + PCA 载荷/解释方差，供三个子图共用。"""
    mean_curves: dict[str, tuple[str, pd.Series]] = {}
    loadings_by_code: dict[str, tuple[str, pd.DataFrame]] = {}
    explained_rows = []
    for item in items:
        panel = review_common_mod.load_factor_panel(item.factor_panel_path)
        cm = term_structure_mod.build_constant_maturity(
            panel, grid_days=cfg.term_grid_days, method=cfg.term_interp_method
        )
        if cm.empty:
            continue
        mean_curves[item.index_code] = (item.name, cm.mean(axis=0, skipna=True))

        _, loadings, explained = term_structure_mod.fit_term_structure_pca(
            cm, n_components=cfg.term_pca_n_components
        )
        if not loadings.empty:
            loadings_by_code[item.index_code] = (item.name, loadings)
        if not explained.empty:
            ev = explained.copy()
            ev.insert(0, "index_code", item.index_code)
            explained_rows.append(ev)

    explained_all = pd.concat(explained_rows, ignore_index=True) if explained_rows else pd.DataFrame()
    return mean_curves, loadings_by_code, explained_all


def _plot_mean_term_structure(ax, grid_days, mean_curves: dict[str, tuple[str, pd.Series]]) -> None:
    # f1_ann_basis 落盘是小数（-0.08 = -8%），乘 100 换算成图上的百分点，
    # 与 y 轴 "(%)" 单位保持一致；不影响下游 PCA（载荷是方向，与输入线性缩放无关）。
    for i, (code, (name, curve)) in enumerate(mean_curves.items()):
        ax.plot(
            grid_days, curve.to_numpy(dtype=float) * 100, marker="o", linewidth=1.8,
            color=_color_for(code, i), label=_label_for(code, name),
        )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("平均期限结构曲线（全样本）")
    ax.set_xlabel("剩余期限（自然日）")
    ax.set_ylabel("年化基差率 (%)")
    ax.legend(fontsize=8)


def _plot_representative_loadings(
    ax, mean_curves: dict[str, tuple[str, pd.Series]], loadings_by_code: dict[str, tuple[str, pd.DataFrame]]
) -> None:
    if not loadings_by_code:
        ax.set_visible(False)
        return
    # 挑选常数期限曲线非空网格点最多（数据覆盖最完整）的指数展示 PCA 载荷。
    rep_code = max(loadings_by_code, key=lambda c: int(mean_curves[c][1].notna().sum()))
    name, loadings = loadings_by_code[rep_code]
    grid_index = [idx for idx in loadings.index if idx != "component_type"]
    grid_vals = [int(str(idx).split("_")[1]) for idx in grid_index]
    component_types = loadings.loc["component_type"] if "component_type" in loadings.index else None
    colors = ("#1b3358", "#4a7ab5", "#c0392b", "#2e8b57")
    for i, pc in enumerate(loadings.columns):
        y = loadings.loc[grid_index, pc].astype(float).to_numpy()
        ctype = component_types[pc] if component_types is not None else pc
        ax.plot(grid_vals, y, marker="o", linewidth=1.8, color=colors[i % len(colors)], label=f"{pc} {ctype}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"PCA 载荷（{_label_for(rep_code, name)}）")
    ax.set_xlabel("剩余期限（自然日）")
    ax.set_ylabel("载荷")
    ax.legend(fontsize=8)


def _plot_explained_variance(ax, mean_curves: dict[str, tuple[str, pd.Series]], explained_all: pd.DataFrame) -> None:
    if explained_all.empty:
        ax.set_visible(False)
        return
    components = sorted(explained_all["component"].unique(), key=lambda x: int(str(x).replace("PC", "")))
    codes = list(mean_curves.keys())
    width = 0.8 / max(len(codes), 1)
    x_base = np.arange(len(components))
    for i, code in enumerate(codes):
        name = mean_curves[code][0]
        sub = explained_all[explained_all["index_code"] == code].set_index("component")
        y = [float(sub.loc[c, "explained_variance_ratio"]) * 100 if c in sub.index else 0.0 for c in components]
        ax.bar(x_base + i * width, y, width=width, color=_color_for(code, i), label=_label_for(code, name))
    ax.set_xticks(x_base + width * (len(codes) - 1) / 2)
    ax.set_xticklabels(components)
    ax.set_ylabel("解释方差占比 (%)")
    ax.set_title("PCA 各主成分解释方差")
    ax.legend(fontsize=8)


def plot_term_structure_chart(
    variant: str,
    items: list[review_common_mod.VariantIndex],
    cfg: config_mod.ReviewConfig,
    out_path: Path,
) -> Optional[Path]:
    mean_curves, loadings_by_code, explained_all = _term_structure_panel_data(items, cfg)
    if not mean_curves:
        logger.warning("[chart_export] %s：没有可用的期限结构数据，跳过画图。", variant)
        return None

    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_mean_term_structure(axes[0], cfg.term_grid_days, mean_curves)
    _plot_representative_loadings(axes[1], mean_curves, loadings_by_code)
    _plot_explained_variance(axes[2], mean_curves, explained_all)

    fig.suptitle(f"期限结构复盘（{variant}）")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("[chart_export] 期限结构图表已保存: %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 基差系数（output/结果图表/基差系数/）
# --------------------------------------------------------------------------- #

def _load_regression_coefs(item: review_common_mod.VariantIndex) -> Optional[pd.DataFrame]:
    path = item.factor_panel_path.parent / "regression_coefs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date") if not df.empty else None


def _plot_coef_series(ax, coefs_by_code: dict[str, tuple[str, pd.DataFrame]], col: str, title: str) -> None:
    plotted = False
    for i, (code, (name, df)) in enumerate(coefs_by_code.items()):
        if col not in df.columns:
            continue
        ax.plot(df["date"], df[col], linewidth=1.2, color=_color_for(code, i), label=_label_for(code, name))
        plotted = True
    if not plotted:
        ax.set_visible(False)
        return
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("系数")
    ax.legend(fontsize=8)


def _plot_significance_share(ax, coefs_by_code: dict[str, tuple[str, pd.DataFrame]]) -> None:
    codes = list(coefs_by_code.keys())
    if not codes:
        ax.set_visible(False)
        return
    a1_share, f3_share = [], []
    for code in codes:
        _, df = coefs_by_code[code]
        a1_col, f3_col = "tstat_f1_ann_basis", "tstat_f3_term_structure"
        a1_share.append(float((df[a1_col].abs() >= 1.96).mean()) if a1_col in df.columns else 0.0)
        f3_share.append(float((df[f3_col].abs() >= 1.96).mean()) if f3_col in df.columns else 0.0)

    x = np.arange(len(codes))
    width = 0.35
    ax.bar(x - width / 2, a1_share, width=width, color="#1b3358", label="a1 (f1_ann_basis) 显著份额")
    ax.bar(x + width / 2, f3_share, width=width, color="#c0392b", label="f3 (期限结构) 显著份额")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_label_for(c, coefs_by_code[c][0]) for c in codes], fontsize=8, rotation=15
    )
    ax.set_ylabel("|t| ≥ 1.96 占比")
    ax.set_title("系数显著性份额（HAC t 检验，walk-forward）")
    ax.legend(fontsize=8)


def plot_basis_coefficient_chart(
    variant: str,
    items: list[review_common_mod.VariantIndex],
    out_path: Path,
) -> Optional[Path]:
    coefs_by_code: dict[str, tuple[str, pd.DataFrame]] = {}
    for item in items:
        df = _load_regression_coefs(item)
        if df is not None:
            coefs_by_code[item.index_code] = (item.name, df)
    if not coefs_by_code:
        logger.warning("[chart_export] %s：没有可用的 regression_coefs.csv，跳过画图。", variant)
        return None

    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_coef_series(axes[0], coefs_by_code, "f1_ann_basis", "当期年化基差系数 a1（walk-forward）")
    _plot_coef_series(axes[1], coefs_by_code, "f3_term_structure", "期限结构系数 f3（walk-forward）")
    _plot_significance_share(axes[2], coefs_by_code)

    fig.suptitle(f"基差回归系数复盘（{variant}）")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("[chart_export] 基差系数图表已保存: %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = config_mod.DEFAULT_REVIEW_CONFIG
    items = review_common_mod.discover_variant_indices()
    if not items:
        logger.warning("output/ 下没有任何 factor_panel.csv，先跑 run.py 再执行本模块。")
        return

    by_variant: dict[str, list[review_common_mod.VariantIndex]] = {}
    for item in items:
        by_variant.setdefault(item.variant, []).append(item)

    for variant, variant_items in by_variant.items():
        plot_term_structure_chart(
            variant, variant_items, cfg, TERM_STRUCTURE_CHART_DIR / f"term_structure_pca_chart_{variant}.png"
        )
        plot_basis_coefficient_chart(
            variant, variant_items, BASIS_COEF_CHART_DIR / f"basis_coefficients_chart_{variant}.png"
        )


if __name__ == "__main__":
    main()
