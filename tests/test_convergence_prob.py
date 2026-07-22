"""convergence_prob.py 的单测：删失口径不泄漏、到期效应分层、分红过滤、分桶顺序。"""

import numpy as np
import pandas as pd
import pytest

from basis_model import config as config_mod
from basis_model import convergence_prob as cp_mod


def _state_panel(n: int = 120, dte_start: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "index_code": "000300",
            "month_tag": "当月",
            "contract": "IF_TEST",
            "f1_ann_basis": np.zeros(n),
            "days_to_expiry": np.arange(dte_start, dte_start - n, -1),
        }
    )


def test_bucket_labels_are_in_ascending_numeric_order():
    labels = cp_mod.bucket_labels((-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    assert labels == [
        "(-inf,-3]", "(-3,-2]", "(-2,-1]", "(-1,0]",
        "(0,1]", "(1,2]", "(2,3]", "(3,inf)",
    ]


def test_label_zscore_bucket_matches_bucket_labels_order():
    z = pd.Series([-10.0, -2.5, -0.5, 0.5, 2.5, 10.0])
    labels = cp_mod.label_zscore_bucket(z)
    assert list(labels) == ["(-inf,-3]", "(-3,-2]", "(-1,0]", "(0,1]", "(2,3]", "(3,inf)"]


def test_censored_tail_rows_are_nan_not_false():
    """历史末尾不足 N 个交易日的观测应该是删失（NaN），不能被当成"未收敛"(False)，
    否则会人为压低经验收敛概率。"""
    n = 50
    dates = pd.bdate_range("2021-01-04", periods=n)
    f1 = np.zeros(n)
    center = np.zeros(n)
    std = np.ones(n) * 0.1
    # f1 恒等于 center：只要没被删失，理应立即判定收敛(converged=True, day=0)。
    events = cp_mod._events_for_group(f1, center, std, dates.to_numpy(), N=10, std_threshold=0.5, decay_threshold=0.5)
    assert events["converged"].iloc[: n - 10].eq(True).all()
    assert events["converged"].iloc[n - 10 :].isna().all()
    assert events["days_to_converge"].iloc[n - 10 :].isna().all()


def test_convergence_uses_only_future_window_not_data_beyond_cutoff_plus_n():
    """修改 t+N 之后（超出该事件窗口）的数据，不应该改变 t 这一天的收敛判定，
    因为窗口本身就该在 t+N 截止——这是本模块"未来窗口不无限扩展"的防前视要求。
    """
    n = 60
    dates = pd.bdate_range("2021-01-04", periods=n).to_numpy()
    center = np.zeros(n)
    std = np.ones(n) * 0.1
    f1 = np.full(n, 5.0)  # 一直没有收敛
    f1[0] = 5.0
    baseline = cp_mod._events_for_group(f1, center, std, dates, N=10, std_threshold=0.5, decay_threshold=0.5)

    modified = f1.copy()
    modified[25] = 0.0  # 远超出 t=0 的 [1,10] 窗口，只应影响 t=15..24 的窗口
    modified_events = cp_mod._events_for_group(modified, center, std, dates, N=10, std_threshold=0.5, decay_threshold=0.5)
    assert baseline["converged"].iloc[0] == modified_events["converged"].iloc[0]
    assert baseline["days_to_converge"].iloc[0] is np.nan or pd.isna(baseline["days_to_converge"].iloc[0])


def test_expiry_layer_filters_short_dte_from_main_sample():
    panel = _state_panel(n=150, dte_start=30)
    state = cp_mod.attach_state(panel, config_mod.DEFAULT_REVIEW_CONFIG)
    events = cp_mod.build_convergence_events(state, n_horizons=(8,))
    main_table = cp_mod.estimate_convergence_table(events, N=8, min_dte=20, exclude_dividend_season=False)
    all_table = cp_mod.estimate_convergence_table(events, N=8, min_dte=None, exclude_dividend_season=False)
    main_n = main_table["n_obs"].sum() if not main_table.empty else 0
    all_n = all_table["n_obs"].sum() if not all_table.empty else 0
    assert main_n <= all_n


def test_estimate_convergence_table_excludes_dividend_season_when_requested():
    n = 200
    dates = pd.bdate_range("2021-01-04", periods=n)
    panel = pd.DataFrame(
        {
            "date": dates,
            "index_code": "000300",
            "month_tag": "当月",
            "contract": "IF_TEST",
            "f1_ann_basis": np.linspace(-0.2, 0.2, n),
            "days_to_expiry": np.full(n, 100),
        }
    )
    state = cp_mod.attach_state(panel, config_mod.DEFAULT_REVIEW_CONFIG)
    events = cp_mod.build_convergence_events(state, n_horizons=(8,))
    with_season = cp_mod.estimate_convergence_table(events, N=8, min_dte=None, exclude_dividend_season=False)
    without_season = cp_mod.estimate_convergence_table(events, N=8, min_dte=None, exclude_dividend_season=True)
    n_with = with_season["n_obs"].sum() if not with_season.empty else 0
    n_without = without_season["n_obs"].sum() if not without_season.empty else 0
    assert n_without < n_with


def test_block_bootstrap_ci_contains_sample_mean_and_is_bounded():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2, size=200).astype(float)
    lo, hi = cp_mod.block_bootstrap_ci(values, block_len=8, reps=200, seed=1)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= values.mean() <= hi


def test_bucket_lookup_table_round_trips_estimate_convergence_table():
    panel = _state_panel(n=300, dte_start=200)
    panel["f1_ann_basis"] = np.sin(np.linspace(0, 20, 300)) * 0.05
    state = cp_mod.attach_state(panel, config_mod.DEFAULT_REVIEW_CONFIG)
    events = cp_mod.build_convergence_events(state, n_horizons=(8,))
    table = cp_mod.estimate_convergence_table(events, N=8, min_dte=None, exclude_dividend_season=False)
    lookup = cp_mod.bucket_lookup_table(table)
    if not table.empty:
        first_bucket = str(table["z_bucket"].iloc[0])
        assert lookup[first_bucket] == pytest.approx(table["p_converge"].iloc[0])
