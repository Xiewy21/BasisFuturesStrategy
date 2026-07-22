"""neutral_hedge_timing.py 的单测：敞口映射的边界行为 + walk-forward 不泄漏。"""

import numpy as np
import pandas as pd
import pytest

from basis_model import config as config_mod
from basis_model import neutral_hedge_timing as nht_mod


def test_compute_richness_maps_z_clip_range_to_unit_interval():
    z = pd.Series([-10.0, -3.0, 0.0, 3.0, 10.0])
    richness = nht_mod.compute_richness(z, z_clip=3.0)
    assert richness.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_map_exposure_matches_named_extreme_cases_from_docstring():
    """复现模块 docstring 里两个具名的极端情形：
    深贴水(richness=0)+高收敛概率 -> 敞口比基础 richness 更低（更坚决地减）；
    升水/贴水浅(richness=1)+高走阔概率(即低p_converge) -> 敞口比基础 richness 更高。
    """
    richness = pd.Series([0.0, 1.0])
    high_converge = pd.Series([0.9, 0.1])  # 第二行 p_converge 低 = 走阔概率高
    exposure = nht_mod.map_exposure(richness, high_converge, tilt_weight=0.4)
    assert exposure.iloc[0] < 0.0 + 1e-9 or exposure.iloc[0] == 0.0  # clip 到 0，已经是最低
    assert exposure.iloc[1] == 1.0  # clip 到 1，已经是最高

    # 换成"不确认"的一侧（p_converge=0.5 中性），敞口应该退化成纯 richness。
    neutral = pd.Series([0.5, 0.5])
    exposure_neutral = nht_mod.map_exposure(richness, neutral, tilt_weight=0.4)
    assert exposure_neutral.tolist() == pytest.approx([0.0, 1.0])


def test_map_exposure_never_flips_direction_relative_to_richness():
    """conviction 只调节幅度，不应该让敞口的相对大小关系反过来
    （richness 更高的一侧，敞口不应该低于 richness 更低的一侧，在同样 conviction 下）。"""
    richness = pd.Series([0.2, 0.8])
    same_conviction = pd.Series([0.7, 0.7])
    exposure = nht_mod.map_exposure(richness, same_conviction, tilt_weight=0.4)
    assert exposure.iloc[0] < exposure.iloc[1]


def test_map_exposure_stays_within_zero_one_bounds():
    rng = np.random.default_rng(0)
    richness = pd.Series(rng.uniform(0, 1, 500))
    p_converge = pd.Series(rng.uniform(0, 1, 500))
    exposure = nht_mod.map_exposure(richness, p_converge, tilt_weight=0.4)
    assert exposure.between(0.0, 1.0).all()


def _synthetic_front_panel(n: int = 900, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    noise = rng.normal(0, 0.02, n)
    ann_basis = -0.05 + np.sin(np.linspace(0, 30, n)) * 0.03 + noise
    S = 4000.0 + np.cumsum(rng.normal(0, 5, n))
    D = 15  # 固定剩余交易日，简化 adj_basis 反推
    adj_basis = ann_basis * S * D / 252.0
    return pd.DataFrame(
        {
            "date": dates,
            "contract": "IF_FRONT",
            "index_code": "000300",
            "month_tag": "当月",
            "F": S + adj_basis,
            "S": S,
            "days_to_expiry": D,
            "calendar_days_to_expiry": D,
            "adj_basis": adj_basis,
            "f1_ann_basis": ann_basis,
        }
    )


def test_walkforward_hedge_timing_has_no_overlap_and_purge_embargo_on_both_sides():
    panel = _synthetic_front_panel(n=900)
    cfg = config_mod.ReviewConfig(
        hedge_timing_train_years=1, hedge_timing_test_months=3,
        hedge_timing_purge_days=5, hedge_timing_embargo_days=5, n_horizon=5,
        basis_center_window=60, basis_center_min_periods=20,
        convergence_n_horizons=(5, 20, 60),
    )
    oos, metrics, log = nht_mod.run_walkforward_hedge_timing(panel, "000300", "raw_basis", cfg)
    assert not oos["date"].duplicated().any()
    for row in log.itertuples(index=False):
        assert row.oos_start_after_embargo > row.train_end_after_purge
        assert (row.oos_start_after_embargo - row.train_end_after_purge).days >= 1


def test_training_probability_table_does_not_use_embargoed_or_test_period_data():
    """把某个窗口测试期(embargo 之后)的 f1_ann_basis 全部改成不可能出现的极端值，
    不应该改变同一个窗口训练期估计出的收敛概率表 —— 这是 purge+embargo 存在的
    意义：训练表只能用 purge 截止日之前的数据估计。
    """
    panel = _synthetic_front_panel(n=900)
    cfg = config_mod.ReviewConfig(
        hedge_timing_train_years=1, hedge_timing_test_months=3,
        hedge_timing_purge_days=5, hedge_timing_embargo_days=5, n_horizon=5,
        basis_center_window=60, basis_center_min_periods=20,
        convergence_n_horizons=(5, 20, 60),
    )
    oos_baseline, _, log_baseline = nht_mod.run_walkforward_hedge_timing(panel, "000300", "raw_basis", cfg)

    corrupted = panel.copy()
    mid_window = log_baseline.iloc[len(log_baseline) // 2]
    test_start = mid_window["oos_start_after_embargo"]
    corrupted.loc[corrupted["date"] >= test_start, "f1_ann_basis"] = 999.0
    corrupted.loc[corrupted["date"] >= test_start, "adj_basis"] = 999.0 * corrupted["S"]

    oos_corrupted, _, log_corrupted = nht_mod.run_walkforward_hedge_timing(
        corrupted, "000300", "raw_basis", cfg
    )
    # 训练期窗口边界与训练期用到的事件数量应该完全一致（因为训练数据没变）。
    pd.testing.assert_frame_equal(
        log_baseline[["window_id", "train_start", "train_end_after_purge", "n_train_events_after_purge"]],
        log_corrupted[["window_id", "train_start", "train_end_after_purge", "n_train_events_after_purge"]],
    )
    # 该窗口测试期之前（含训练+purge+embargo 边界之前）的敞口应完全不受污染影响。
    before = oos_baseline[oos_baseline["date"] < test_start].reset_index(drop=True)
    before_corrupted = oos_corrupted[oos_corrupted["date"] < test_start].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, before_corrupted, check_dtype=False)


def test_cost_saving_uses_non_overlapping_single_day_returns():
    """daily_cost 必须用单日 ret_1d_basis（非重叠），沿用
    strategy.backtest_position_series 同一条纪律，不能用重叠 N 日收益。"""
    panel = _synthetic_front_panel(n=900)
    cfg = config_mod.ReviewConfig(
        hedge_timing_train_years=1, hedge_timing_test_months=3,
        hedge_timing_purge_days=5, hedge_timing_embargo_days=5, n_horizon=5,
        basis_center_window=60, basis_center_min_periods=20,
        convergence_n_horizons=(5, 20, 60),
    )
    oos, metrics, log = nht_mod.run_walkforward_hedge_timing(panel, "000300", "raw_basis", cfg)
    expected_dynamic = oos["exposure"] * oos["ret_1d_basis"]
    assert oos["dynamic_daily_cost"].equals(expected_dynamic)
    assert oos["benchmark_daily_cost"].equals(oos["ret_1d_basis"])
