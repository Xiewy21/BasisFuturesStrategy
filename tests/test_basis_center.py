"""basis_center.py 的单测：因果滚动不泄漏未来、异常口径独立、持续性口径正确、
新增的滚动分位 p(t) 与常数期限口径中枢。
"""

import numpy as np
import pandas as pd
import pytest

from basis_model import basis_center as bc_mod
from basis_model import config as config_mod


def _synthetic_series(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    f1 = pd.Series(rng.normal(-0.05, 0.02, n)).cumsum() * 0.0 + rng.normal(-0.05, 0.03, n)
    return pd.DataFrame(
        {
            "date": dates,
            "index_code": ["000300"] * n,
            "month_tag": ["当月"] * n,
            "f1_ann_basis": f1,
        }
    )


def _synthetic_four_contract_panel(n_days: int = 400, seed: int = 2) -> pd.DataFrame:
    """构造一个含四个 month_tag、且带 calendar_days_to_expiry 的完整面板，
    供需要跑常数期限插值（build_constant_maturity）的端到端测试使用。
    """
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    rng = np.random.default_rng(seed)
    rows = []
    tag_calendar_dte = {"当月": 20, "次月": 70, "季月": 150, "下季月": 220}
    tag_level = {"当月": -0.10, "次月": -0.08, "季月": -0.05, "下季月": -0.02}
    for d in dates:
        for tag, dte in tag_calendar_dte.items():
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
                    "f1_ann_basis": tag_level[tag] + rng.normal(0, 0.002),
                }
            )
    return pd.DataFrame(rows)


def test_causal_rolling_stats_center_excludes_current_day_value():
    """把某一天的观测改成极端异常值，只应影响它之后的滚动窗口，不能影响
    它自己那一天的 center/std（因为 shift(1) 保证当天值不进当天窗口）。
    """
    df = _synthetic_series(n=200)
    baseline = bc_mod.causal_rolling_stats(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"],
        window=50, min_periods=10,
    )
    shocked = df.copy()
    shock_idx = 100
    shocked.loc[shock_idx, "f1_ann_basis"] = 999.0
    shocked_stats = bc_mod.causal_rolling_stats(
        shocked["f1_ann_basis"], shocked["index_code"], shocked["month_tag"], shocked["date"],
        window=50, min_periods=10,
    )
    # 冲击当天自己的 center/std 不应该变化（因为该值被 shift(1) 排除在自己窗口外）。
    assert baseline.loc[shock_idx, "center"] == pytest.approx(shocked_stats.loc[shock_idx, "center"])
    # 冲击后一天的 center 必须变化（现在冲击值进入了它的滚动窗口）。
    assert baseline.loc[shock_idx + 1, "center"] != pytest.approx(
        shocked_stats.loc[shock_idx + 1, "center"]
    )
    # 冲击之前的所有 center 完全不受未来冲击影响。
    pd.testing.assert_series_equal(
        baseline.loc[: shock_idx - 1, "center"], shocked_stats.loc[: shock_idx - 1, "center"]
    )


def test_causal_rolling_stats_grouped_by_index_and_month_tag():
    """两个不同 (index_code, period) 分组必须各自独立滚动，互不借用对方历史。"""
    n = 100
    dates = pd.bdate_range("2020-01-02", periods=n)
    df = pd.concat(
        [
            pd.DataFrame({"date": dates, "index_code": "000300", "month_tag": "当月", "f1_ann_basis": 0.0}),
            pd.DataFrame({"date": dates, "index_code": "000300", "month_tag": "次月", "f1_ann_basis": 1.0}),
        ],
        ignore_index=True,
    )
    stats = bc_mod.causal_rolling_stats(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"],
        window=20, min_periods=5,
    )
    front = stats[df["month_tag"] == "当月"]["center"].dropna()
    second = stats[df["month_tag"] == "次月"]["center"].dropna()
    assert (front == 0.0).all()
    assert (second == 1.0).all()


def test_z_score_uses_causal_center_and_std():
    df = _synthetic_series(n=200)
    stats = bc_mod.causal_rolling_stats(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"], window=50, min_periods=10,
    )
    z = bc_mod.compute_z_score(df["f1_ann_basis"], stats["center"], stats["std"])
    manual = (df["f1_ann_basis"] - stats["center"]) / stats["std"]
    pd.testing.assert_series_equal(z, manual, check_names=False)


def test_flag_anomalies_persistence_only_confirms_at_mth_day():
    """构造一段连续 5 天位于极端分位以上的序列，persistence_m=3 时前 2 天不应
    标记为持续异常，第 3 天开始才标记（因果定义：走到第 m 天才承认"进入持续"）。
    """
    n = 60
    dates = pd.bdate_range("2020-01-02", periods=n)
    values = np.zeros(n)
    # 前 40 天固定为常数 0（而非带噪声的序列），确保滚动分位基准本身不会
    # 因为随机噪声偶然越界，产生和尖峰无关的"假阳性"游程，干扰断言。
    # 后 5 天连续大幅拉升，制造持续性异常。
    spike_start = 40
    values[spike_start : spike_start + 5] = 5.0
    df = pd.DataFrame(
        {"date": dates, "index_code": "000300", "month_tag": "当月", "f1_ann_basis": values}
    )
    stats = bc_mod.causal_rolling_stats(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"],
        window=30, min_periods=10, quantiles=(0.05, 0.95),
    )
    z = bc_mod.compute_z_score(df["f1_ann_basis"], stats["center"], stats["std"])
    anomalies = bc_mod.flag_anomalies(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"], stats, z,
        tail_pct=0.05, z_threshold=2.0, persistence_m=3,
    )
    spike_flags = anomalies.loc[spike_start : spike_start + 4, "is_persistent_anomaly"].to_numpy()
    spike_types = anomalies.loc[spike_start : spike_start + 4, "anomaly_type"].to_numpy()
    # 冲击的第 1、2 天（游程长度 1、2）还不足 m=3，不应标记持续异常，但仍是统计异常。
    assert not spike_flags[0]
    assert not spike_flags[1]
    assert spike_types[0] == "statistical"
    assert spike_types[1] == "statistical"
    # 第 3 天起（游程长度达到 3）才标记为 persistent。
    assert spike_flags[2]
    assert spike_flags[3]
    assert spike_flags[4]
    assert spike_types[2] == "persistent"


def test_flag_anomalies_dividend_season_flag_is_independent_of_statistical_flags():
    df = _synthetic_series(n=250)
    stats = bc_mod.causal_rolling_stats(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"], window=60, min_periods=10,
        quantiles=(0.05, 0.95),
    )
    z = bc_mod.compute_z_score(df["f1_ann_basis"], stats["center"], stats["std"])
    anomalies = bc_mod.flag_anomalies(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"], stats, z,
    )
    expected_season = df["date"].dt.month.isin([5, 6, 7])
    pd.testing.assert_series_equal(
        anomalies["is_dividend_season"].reset_index(drop=True),
        expected_season.reset_index(drop=True),
        check_names=False,
    )


def test_long_term_center_is_full_sample_median_not_causal():
    """长期中枢是全样本统计量（明确非因果），用来对照，不应等于任何一天的滚动中枢。"""
    df = _synthetic_series(n=200)
    long_term = bc_mod.long_term_center(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"]
    )
    assert (long_term["center_long_full_sample"] == df["f1_ann_basis"].median()).all()


def test_causal_rolling_percentile_matches_manual_rank_and_excludes_current_value():
    """p(t) 手算校验：today 的值不参与"比较基准"，只作为被比较对象；把 today 的
    值改成极端值不应该影响它自己的 p(t) 定义方式的正确性（能被正确算出接近 1）。
    """
    n = 80
    dates = pd.bdate_range("2021-01-04", periods=n)
    values = np.arange(n, dtype=float)  # 严格递增，任意一天在此前历史中都是新高
    df = pd.DataFrame(
        {"date": dates, "index_code": "000300", "month_tag": "当月", "f1_ann_basis": values}
    )
    p = bc_mod.causal_rolling_percentile(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"],
        window=20, min_periods=20,
    )
    # 递增序列：一旦窗口攒够 20 个历史点，当天的值必然是历史最大值，p(t) 应该恒为 1.0。
    valid = p.dropna()
    assert (valid == 1.0).all()

    # 手算单点：取序列中间某一天，直接算 (history<current) 的比例，核对与函数一致。
    t = 50
    history = values[t - 20 : t]
    current = values[t]
    expected = float(np.mean(history < current))
    assert p.iloc[t] == pytest.approx(expected)


def test_causal_rolling_percentile_future_values_do_not_change_past_p():
    """把某天之后的历史值改成极端值，不应该改变它之前任何一天的 p(t)。"""
    n = 100
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2021-01-04", periods=n)
    values = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {"date": dates, "index_code": "000300", "month_tag": "当月", "f1_ann_basis": values}
    )
    baseline = bc_mod.causal_rolling_percentile(
        df["f1_ann_basis"], df["index_code"], df["month_tag"], df["date"], window=20, min_periods=20,
    )
    corrupted = df.copy()
    cutoff = 60
    corrupted.loc[cutoff:, "f1_ann_basis"] = 999.0
    corrupted_p = bc_mod.causal_rolling_percentile(
        corrupted["f1_ann_basis"], corrupted["index_code"], corrupted["month_tag"], corrupted["date"],
        window=20, min_periods=20,
    )
    pd.testing.assert_series_equal(
        baseline.loc[: cutoff - 1], corrupted_p.loc[: cutoff - 1]
    )


def test_build_basis_center_table_end_to_end_on_synthetic_panel():
    panel = _synthetic_four_contract_panel(n_days=400)
    cfg = config_mod.ReviewConfig(basis_center_window=50, basis_center_window_alt=100, basis_center_min_periods=10)
    table = bc_mod.build_basis_center_table(panel, cfg)
    required_cols = {
        "period", "period_type", "center_250", "std_250", "q25_250", "q50_250", "q75_250",
        "p_250", "p_500", "center_500", "std_500", "z_250", "z_500",
        "center_long_full_sample", "center_long_annual",
        "is_extreme_quantile", "is_extreme_z", "is_statistical_anomaly",
        "run_length", "is_persistent_anomaly", "anomaly_type", "is_dividend_season",
    }
    assert required_cols <= set(table.columns)

    # 标签口径：一行 = 一个 (date, contract)，与原始面板行数一致。
    label_rows = table[table["period_type"] == "label"]
    assert len(label_rows) == len(panel)
    assert set(label_rows["period"].unique()) == {"当月", "次月", "季月", "下季月"}

    # 常数期限口径：一行 = 一个 (date, cm_<grid>)，网格取自 cfg.term_grid_days。
    cm_rows = table[table["period_type"] == "constant_maturity"]
    expected_cm_periods = {f"cm_{d}" for d in cfg.term_grid_days}
    assert set(cm_rows["period"].unique()) == expected_cm_periods
    assert cm_rows["contract"].isna().all()


def test_build_basis_center_table_rejects_multiple_index_codes():
    panel = _synthetic_four_contract_panel(n_days=50)
    other = panel.copy()
    other["index_code"] = "000905"
    mixed = pd.concat([panel, other], ignore_index=True)
    with pytest.raises(ValueError):
        bc_mod.build_basis_center_table(mixed, config_mod.DEFAULT_REVIEW_CONFIG)
