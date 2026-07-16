"""防前视（纪律 2）与硬软分离（纪律 3）的单测。

核心断言：
    1. factors.assert_no_lookahead 能拦截训练专用列混入预测路径。
    2. forecast_basis 的函数签名本身就不接受 future_return/y 之类的参数
       （Python 会在调用时直接 TypeError，是最强的防前视手段）。
    3. build_factor_panel 产出的 f2_future_return / y_convergence_rate 在每个
       合约末尾的 N 行必须是 NaN —— 因为那些未来观测还没发生，不能被"未来函数"
       悄悄补出来（例如误用 fillna/bfill）。
    4. forecast_basis 的 from_regression 输出不随 expected_return 情景变化，
       只有 from_qualitative 会变 —— 验证硬软分离没有被打通。
"""

import numpy as np
import pandas as pd
import pytest

from basis_model import factors as factors_mod
from basis_model import forecast as forecast_mod


def test_assert_no_lookahead_blocks_forbidden_columns_dict():
    with pytest.raises(AssertionError):
        factors_mod.assert_no_lookahead({"f2_future_return": 0.01, "other": 1})

    with pytest.raises(AssertionError):
        factors_mod.assert_no_lookahead({"y_convergence_rate": 0.01})

    # 不含禁用字段应该正常通过，不抛异常
    factors_mod.assert_no_lookahead({"f1_ann_basis": -0.05, "f3_term_structure": 0.01})


def test_assert_no_lookahead_blocks_forbidden_columns_dataframe():
    df = pd.DataFrame({"f1_ann_basis": [1.0], "y_convergence_rate": [0.02]})
    with pytest.raises(AssertionError):
        factors_mod.assert_no_lookahead(df)


def test_forecast_basis_signature_rejects_future_return_kwarg():
    """forecast_basis 根本没有 future_return/y_convergence_rate 形参，
    传入这类关键字参数必须在调用层就报错（TypeError），而不是被悄悄接受。
    """
    coef_row = pd.Series({"const": 0.0, "f1_ann_basis": -1.5, "f3_term_structure": 0.0})
    base_kwargs = dict(
        date=pd.Timestamp("2024-06-01"),
        contract="IF2409",
        index_code="000300",
        S=4000.0,
        current_adj_basis=-20.0,
        f1_ann_basis=-0.05,
        f3_term_structure=0.0,
        coef_row=coef_row,
        N=8,
    )
    with pytest.raises(TypeError):
        forecast_mod.forecast_basis(**base_kwargs, future_return=0.03)
    with pytest.raises(TypeError):
        forecast_mod.forecast_basis(**base_kwargs, y_convergence_rate=0.01)


def test_forecast_basis_extra_context_leakage_is_caught():
    coef_row = pd.Series({"const": 0.0, "f1_ann_basis": -1.5, "f3_term_structure": 0.0})
    with pytest.raises(AssertionError):
        forecast_mod.forecast_basis(
            date=pd.Timestamp("2024-06-01"),
            contract="IF2409",
            index_code="000300",
            S=4000.0,
            current_adj_basis=-20.0,
            f1_ann_basis=-0.05,
            f3_term_structure=0.0,
            coef_row=coef_row,
            N=8,
            extra_context={"y_convergence_rate": 0.02},
        )


def test_build_factor_panel_last_n_rows_per_contract_are_nan():
    dates = pd.bdate_range("2024-01-01", periods=30)
    n = 8
    panel = pd.DataFrame(
        {
            "date": dates,
            "contract": ["IF2403"] * 30,
            "index_code": ["000300"] * 30,
            "month_tag": ["当月"] * 30,
            "F": np.linspace(3980, 4010, 30),
            "S": np.linspace(4000, 4000, 30),
            "days_to_expiry": np.arange(60, 30, -1),
            "adj_basis": np.linspace(-20, 10, 30),
        }
    )
    factor_panel = factors_mod.build_factor_panel(panel, N=n)
    sorted_panel = factor_panel.sort_values("date")

    tail = sorted_panel.tail(n)
    assert tail["f2_future_return"].isna().all()
    assert tail["y_convergence_rate"].isna().all()

    head = sorted_panel.head(len(sorted_panel) - n)
    assert head["f2_future_return"].notna().all()
    assert head["y_convergence_rate"].notna().all()


def test_conditional_regression_changes_with_return_scenario_but_behavior_does_not():
    """expected_return 是条件情景，应经由历史估计的月份交互项进入硬回归。"""
    coef_row = pd.Series(
        {
            "const": 0.001,
            "f1_ann_basis": -1.5082,
            "f3_term_structure": 0.0,
            "X_return_当月": 0.5785,
        }
    )
    results = []
    for er in (-0.04, -0.02, 0.0, 0.02, 0.04):
        r = forecast_mod.forecast_basis(
            date=pd.Timestamp("2024-06-01"),
            contract="IF2409",
            index_code="000300",
            S=4000.0,
            current_adj_basis=-20.0,
            f1_ann_basis=-0.05,
            f3_term_structure=0.0,
            coef_row=coef_row,
            N=8,
            expected_return=er,
            month_tag="当月",
        )
        results.append(r)

    hard_points = {r["from_regression"]["predicted_basis_change_points"] for r in results}
    assert len(hard_points) == len(results)

    soft_points = {r["from_qualitative"]["total_qualitative_points"] for r in results}
    assert len(soft_points) == 1


def test_factor_panel_contains_report_return_month_interactions():
    dates = pd.bdate_range("2024-01-01", periods=20)
    panel = pd.DataFrame(
        {
            "date": dates,
            "contract": ["IF2403"] * 20,
            "index_code": ["000300"] * 20,
            "month_tag": ["当月"] * 20,
            "F": np.linspace(3980, 3990, 20),
            "S": np.linspace(4000, 4020, 20),
            "days_to_expiry": np.arange(40, 20, -1),
            "adj_basis": np.linspace(-20, -30, 20),
        }
    )
    out = factors_mod.build_factor_panel(panel, N=5)
    assert {f"X_return_{tag}" for tag in ("当月", "次月", "季月", "下季月")} <= set(out.columns)
    assert out["X_return_当月"].equals(out["f2_future_return"])
