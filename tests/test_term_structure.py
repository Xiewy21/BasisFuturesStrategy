"""term_structure.py 的单测：常数期限插值不外推、PCA 形状分类、回归对比不前视。"""

import numpy as np
import pandas as pd
import pytest

from basis_model import config as config_mod
from basis_model import term_structure as ts_mod


def _synthetic_panel(n_days: int = 40) -> pd.DataFrame:
    """构造一个简单的四合约面板：每天四个 month_tag，days_to_expiry 固定为
    30/70/150/220 个自然日（近似当月/次月/季月/下季月的典型剩余期限），
    f1_ann_basis 按期限做出明显不同的水平，便于断言插值结果落在合理范围内。
    """
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    tag_calendar_dte = {"当月": 20, "次月": 70, "季月": 150, "下季月": 220}
    tag_level = {"当月": -0.10, "次月": -0.08, "季月": -0.05, "下季月": -0.02}
    rng = np.random.default_rng(0)
    for i, d in enumerate(dates):
        for tag, dte in tag_calendar_dte.items():
            noise = rng.normal(0, 0.001)
            rows.append(
                {
                    "date": d,
                    "contract": f"IF_{tag}",
                    "index_code": "000300",
                    "month_tag": tag,
                    "F": 4000.0,
                    "S": 4000.0,
                    "days_to_expiry": int(dte * 0.7),
                    "calendar_days_to_expiry": dte,
                    "adj_basis": 0.0,
                    "f1_ann_basis": tag_level[tag] + noise,
                    "f3_term_structure": tag_level["季月"] - tag_level["下季月"],
                    "y_convergence_rate": rng.normal(0, 0.01),
                    "D_month_季月": float(tag == "季月"),
                    "D_month_当月": float(tag == "当月"),
                    "D_month_次月": float(tag == "次月"),
                }
            )
    return pd.DataFrame(rows)


def test_constant_maturity_no_extrapolation_beyond_grid():
    """网格点 [30,60,90,180] 全部落在当天 [20,220] 的观测范围内时应有值；
    若某天的观测范围收窄到落不进某个网格点，该格必须是 NaN，不能外推。"""
    panel = _synthetic_panel()
    cm = ts_mod.build_constant_maturity(panel, grid_days=config_mod.TERM_GRID_DAYS)
    assert cm.shape[1] == 4
    # 观测范围 [20,220] 完整覆盖 [30,60,90,180]，因此不应该有 NaN。
    assert cm.notna().all().all()

    # 收窄单日观测范围到 [20,90]，180 这个格点应该变成 NaN（超出当天最大剩余期限）。
    narrowed = panel.copy()
    one_day = narrowed["date"].iloc[0]
    narrowed.loc[narrowed["date"] == one_day, "calendar_days_to_expiry"] = [20, 40, 60, 90]
    cm2 = ts_mod.build_constant_maturity(narrowed, grid_days=config_mod.TERM_GRID_DAYS)
    assert pd.isna(cm2.loc[one_day, "cm_180"])
    assert cm2.loc[one_day, ["cm_30", "cm_60", "cm_90"]].notna().all()


def test_constant_maturity_linear_interpolation_matches_manual_calc():
    """用两点手算验证线性插值公式，避免插值实现本身算错方向或权重。"""
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "calendar_days_to_expiry": [20.0, 100.0],
            "f1_ann_basis": [-0.10, -0.02],
        }
    )
    cm = ts_mod.build_constant_maturity(panel, grid_days=(60,), method="linear")
    # 60 在 [20,100] 中点偏左：手算线性插值 = -0.10 + (60-20)/(100-20)*( -0.02-(-0.10))
    expected = -0.10 + (60 - 20) / (100 - 20) * (-0.02 - (-0.10))
    assert cm.loc[pd.Timestamp("2024-01-02"), "cm_60"] == pytest.approx(expected)


def test_pca_classifies_level_slope_curvature_by_sign_changes():
    assert ts_mod._classify_component(np.array([0.5, 0.4, 0.3, 0.2])) == "level"
    assert ts_mod._classify_component(np.array([0.6, 0.1, -0.2, -0.5])) == "slope"
    assert ts_mod._classify_component(np.array([-0.3, 0.6, 0.1, -0.6])) == "curvature"


def test_pca_scores_only_use_complete_rows_and_recover_known_variance_order():
    panel = _synthetic_panel(n_days=80)
    cm = ts_mod.build_constant_maturity(panel, grid_days=config_mod.TERM_GRID_DAYS)
    scores, loadings, explained = ts_mod.fit_term_structure_pca(cm, n_components=3)
    # 完整样本日数应等于常数期限表里没有任何 NaN 的行数。
    assert len(scores) == len(cm.dropna(how="any"))
    # 方差解释比例降序（PCA 按定义应满足）。
    ratios = explained["explained_variance_ratio"].to_numpy()
    assert (np.diff(ratios) <= 1e-9).all()
    assert explained["explained_variance_ratio"].sum() <= 1.0 + 1e-9


def test_compare_pca_vs_f3_does_not_leak_future_rows_into_early_windows():
    """回归对比复用 regression.rolling_ols，其因果性已由既有测试覆盖；这里额外
    断言：把某个截止日之后的 y/f1 全部替换成极端异常值，不会改变该截止日之前
    已经产出的系数（因为 rolling_ols 只应使用 date<=cutoff 的窗口）。
    """
    panel = _synthetic_panel(n_days=400)
    cm = ts_mod.build_constant_maturity(panel, grid_days=config_mod.TERM_GRID_DAYS)
    scores, loadings, explained = ts_mod.fit_term_structure_pca(cm, n_components=3)
    scores_named = ts_mod._rename_scores_by_type(scores, explained)
    assert {"PC_slope", "PC_curv"} <= set(scores_named.columns)

    cfg = config_mod.ReviewConfig(rolling_window=60, min_obs=20, refit_every=5, n_horizon=5)
    baseline = ts_mod.compare_pca_vs_f3(panel, scores_named, cfg)
    early_cutoff = baseline["date"].iloc[len(baseline) // 4]

    corrupted = panel.copy()
    corrupted.loc[corrupted["date"] > early_cutoff, "f1_ann_basis"] = 999.0
    corrupted.loc[corrupted["date"] > early_cutoff, "y_convergence_rate"] = 999.0
    rerun = ts_mod.compare_pca_vs_f3(corrupted, scores_named, cfg)

    before = baseline[baseline["date"] <= early_cutoff].reset_index(drop=True)
    before_rerun = rerun[rerun["date"] <= early_cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, before_rerun)


def test_run_for_panel_handles_insufficient_complete_rows_gracefully():
    """常数期限完整行数不够拟合 PCA 时（比如历史极短），不应抛异常，应返回空表。"""
    panel = _synthetic_panel(n_days=2)
    results = ts_mod.run_for_panel(panel, config_mod.ReviewConfig())
    assert results["pca_scores"].empty
    assert results["regression_comparison"].empty
