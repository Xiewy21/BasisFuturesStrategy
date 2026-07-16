"""十年期理论回测新增能力的单测：config.py 的基本合法性、MockAdapter 的
"当月"标签不会指向已到期合约（曾经的真实 bug，由 hedge_cost_ladder 意外
暴露）、regression.rolling_ols 的 Newey-West 显著性列、strategy.hedge_cost_ladder
的基本形状。
"""

from __future__ import annotations

import pandas as pd
import pytest

from basis_model import config as config_mod
from basis_model import data as data_mod
from basis_model import dividend as dividend_mod
from basis_model import factors as factors_mod
from basis_model import regression as regression_mod
from basis_model import strategy as strategy_mod


def test_index_configs_cover_four_products_with_valid_dates():
    products = {c.product for c in config_mod.INDEX_CONFIGS}
    assert products == {"IH", "IF", "IC", "IM"}
    for cfg in config_mod.INDEX_CONFIGS:
        # data_start 必须是合法的 ISO 日期字符串
        pd.Timestamp(cfg.data_start)


def test_im_has_shorter_history_note_others_do_not():
    by_product = {c.product: c for c in config_mod.INDEX_CONFIGS}
    assert by_product["IM"].short_history_note is not None
    assert by_product["IH"].short_history_note is None
    assert by_product["IC"].short_history_note is None
    # IM 的起始日应晚于全局 START_DATE
    assert pd.Timestamp(by_product["IM"].data_start) > pd.Timestamp(config_mod.START_DATE)


def test_all_n_values_dedups_and_leads_with_primary_n():
    cfg = config_mod.BacktestConfig(n_horizon=8, n_robustness=(8, 5, 10, 20))
    assert cfg.all_n_values == [8, 5, 10, 20]  # 主 N 在前，且去重


def test_default_window_is_a_rolling_ten_year_period():
    start = pd.Timestamp(config_mod.START_DATE)
    end = pd.Timestamp(config_mod.END_DATE)
    assert end.year - start.year == 10
    assert (end.month, end.day) == (start.month, start.day)


def test_mock_front_month_never_points_to_an_expired_contract():
    """回归测试：MockAdapter 曾经在每月第三个周五之后，把已到期合约继续标成
    "当月"，导致该行被 days_to_expiry<=0 过滤掉，"当月"标签系统性缺失约 40%
    的观测。这里直接跑一整年，断言"当月"和其他标签的行数应当基本相当。
    """
    mock = data_mod.MockAdapter()
    panel = mock.generate_mock_panel("000300", "2024-01-01", "2024-12-31")
    counts = panel["month_tag"].value_counts()

    assert set(counts.index) == {"当月", "次月", "季月", "下季月"}
    # 四个标签理论上应该每天各出现一次，允许极小的边界误差
    max_count = counts.max()
    min_count = counts.min()
    assert min_count / max_count > 0.95


def test_mock_panel_is_reproducible_and_neutral():
    """分别加载期货和现货时路径必须一致，且不能硬编码极端贴水先验。"""
    mock = data_mod.MockAdapter(seed=42)
    first = mock.generate_mock_panel("000300", "2024-01-01", "2024-12-31")
    second = mock.generate_mock_panel("000300", "2024-01-01", "2024-12-31")
    pd.testing.assert_frame_equal(first, second)

    annualized = (
        first["F"].sub(first["S"]).div(first["S"])
        .mul(252).div(first["days_to_expiry"])
    )
    assert annualized.abs().max() < 0.05


def test_rolling_ols_with_hac_lags_adds_significance_columns():
    dates = pd.bdate_range("2024-01-01", periods=120)
    n = len(dates)
    rng = pd.Series(range(n)).astype(float)
    noise = (rng * 37 % 7 - 3) * 0.02  # 确定性小扰动，避免完全共线导致的除零警告
    panel = pd.DataFrame(
        {
            "date": dates,
            "contract": ["IF2409"] * n,
            "index_code": ["000300"] * n,
            "month_tag": ["季月"] * n,
            "F": 3980.0 + rng * 0.1,
            "S": 4000.0,
            "days_to_expiry": list(range(200, 200 - n, -1)),
            "adj_basis": -20.0 + rng * 0.05 + noise,
        }
    )
    factor_panel = factors_mod.build_factor_panel(panel, N=5)
    # 单一合约、单一月份标签的合成数据没有"下季月"对照，f3_term_structure 会是
    # 全 NaN；这里只关心 HAC 显著性列的产出机制，因此只用 f1 作为自变量。
    coefs = regression_mod.rolling_ols(
        factor_panel,
        window=60,
        min_obs=20,
        step=5,
        hac_lags=5,
        feature_cols=("f1_ann_basis",),
        include_month_dummies=False,
    )

    assert not coefs.empty
    assert "tstat_f1_ann_basis" in coefs.columns
    assert "pvalue_f1_ann_basis" in coefs.columns
    assert coefs["tstat_f1_ann_basis"].notna().any()


def test_hedge_cost_ladder_returns_four_ranked_rules():
    mock = data_mod.MockAdapter()
    panel = mock.generate_mock_panel("000905", "2023-01-01", "2023-12-31")
    panel_adj = dividend_mod.dividend_adjust_panel(panel, assume_zero_dividend=True)
    factor_panel = factors_mod.build_factor_panel(panel_adj, N=8)
    factor_panel = factor_panel.sort_values(["contract", "date"]).reset_index(drop=True)
    factor_panel["hedge_cost"] = strategy_mod.hedge_cost(
        factor_panel["adj_basis"], factor_panel["contract"], factor_panel["date"], 8
    )
    coefs = regression_mod.rolling_ols(factor_panel, window=60, min_obs=20, step=5, hac_lags=8)
    all_dates = pd.Index(sorted(factor_panel["date"].unique()), name="date")
    coefs_daily = coefs.reindex(all_dates).ffill()

    ladder = strategy_mod.hedge_cost_ladder(factor_panel, coefs_daily, N=8)

    assert set(ladder["策略"]) == {"当月", "季月", "最低贴水", "回归展仓"}
    assert ladder["mean_annualized_cost"].is_monotonic_increasing
    assert (ladder["n_common_evaluation_dates"] == ladder["n_common_evaluation_dates"].iloc[0]).all()
    events = ladder.attrs["roll_events"]
    assert {"date", "old_contract", "new_contract", "F_old", "F_new", "S",
            "interval_days", "points", "pct", "annualized", "策略"}.issubset(events.columns)
    assert (events["points"] == events["F_old"] - events["F_new"]).all()
