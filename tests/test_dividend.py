"""分红调整基差的单测：验证核心恒等式与除息事件窗口边界。"""

import pandas as pd
import pytest

from basis_model.dividend import (
    div_points,
    dividend_adjust_panel,
    dividend_adjust_panel_point_in_time,
    dividend_adjusted_basis,
    point_in_time_div_points,
)


def test_dividend_adjusted_basis_identity():
    assert dividend_adjusted_basis(100.0, 95.0, 2.0) == 7.0
    assert dividend_adjusted_basis(100.0, 95.0, 0.0) == 5.0


def test_dividend_adjusted_basis_vectorized():
    F = pd.Series([100.0, 200.0])
    S = pd.Series([95.0, 205.0])
    div = pd.Series([1.0, 0.0])
    result = dividend_adjusted_basis(F, S, div)
    assert result.tolist() == [6.0, -5.0]


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["A", "B", "C"],
            "index_code": ["000300", "000300", "000300"],
            "ex_date": [
                pd.Timestamp("2024-01-01"),  # 早于估值日，应被排除
                pd.Timestamp("2024-06-15"),  # 窗口内，应被计入
                pd.Timestamp("2024-10-01"),  # 晚于到期日，应被排除
            ],
            "dividend_per_share": [1.0, 0.5, 2.0],
            "weight_pct": [1.0, 2.0, 1.0],
            "dividend_yield": [0.05, 0.02, 0.03],
        }
    )


def test_div_points_only_counts_events_within_window():
    events = _sample_events()
    valuation_date = pd.Timestamp("2024-03-01")
    expiry_date = pd.Timestamp("2024-09-20")  # 覆盖 06-15，排除 01-01 与 10-01

    points = div_points(
        valuation_date=valuation_date,
        expiry_date=expiry_date,
        index_code="000300",
        dividend_events=events,
        index_level=4000.0,
        assume_zero_dividend=False,
    )
    expected = 0.02 * 2.0 / 100.0 * 4000.0  # 只有 B 那一条落在窗口内
    assert points == pytest.approx(expected)


def test_div_points_excludes_event_exactly_on_valuation_date():
    events = pd.DataFrame(
        {
            "stock_code": ["A"],
            "index_code": ["000300"],
            "ex_date": [pd.Timestamp("2024-03-01")],
            "dividend_per_share": [1.0],
            "weight_pct": [1.0],
            "dividend_yield": [0.05],
        }
    )
    points = div_points(
        valuation_date=pd.Timestamp("2024-03-01"),
        expiry_date=pd.Timestamp("2024-09-20"),
        index_code="000300",
        dividend_events=events,
        index_level=4000.0,
        assume_zero_dividend=False,
    )
    assert points == 0.0  # ex_date == valuation_date 不计入（严格大于）


def test_div_points_includes_event_exactly_on_expiry_date():
    events = pd.DataFrame(
        {
            "stock_code": ["A"],
            "index_code": ["000300"],
            "ex_date": [pd.Timestamp("2024-09-20")],
            "dividend_per_share": [1.0],
            "weight_pct": [1.0],
            "dividend_yield": [0.05],
        }
    )
    points = div_points(
        valuation_date=pd.Timestamp("2024-03-01"),
        expiry_date=pd.Timestamp("2024-09-20"),
        index_code="000300",
        dividend_events=events,
        index_level=4000.0,
        assume_zero_dividend=False,
    )
    expected = 0.05 * 1.0 / 100.0 * 4000.0
    assert points == pytest.approx(expected)


def test_assume_zero_dividend_forces_zero_regardless_of_events():
    events = _sample_events()
    points = div_points(
        valuation_date=pd.Timestamp("2024-03-01"),
        expiry_date=pd.Timestamp("2024-09-20"),
        index_code="000300",
        dividend_events=events,
        index_level=4000.0,
        assume_zero_dividend=True,
    )
    assert points == 0.0


def test_dividend_adjust_panel_default_zero_dividend():
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-03-01")],
            "contract": ["IF2409"],
            "index_code": ["000300"],
            "month_tag": ["季月"],
            "F": [3980.0],
            "S": [4000.0],
            "days_to_expiry": [200],
        }
    )
    out = dividend_adjust_panel(panel)  # assume_zero_dividend 默认 True
    assert out["div_points"].iloc[0] == 0.0
    assert out["adj_basis"].iloc[0] == -20.0


def _pit_inputs(ann_date="2024-04-01"):
    events = pd.DataFrame({
        "stock_code": ["600000.SH"], "index_code": ["000300"],
        "ann_date": [pd.Timestamp(ann_date)], "ex_date": [pd.Timestamp("2024-06-15")],
        "end_date": [pd.Timestamp("2023-12-31")], "cash_div": [1.0],
    })
    weights = pd.DataFrame({
        "stock_code": ["600000.SH", "600000.SH"],
        "trade_date": pd.to_datetime(["2024-01-31", "2024-05-31"]),
        "weight_pct": [2.0, 99.0],
    })
    prices = pd.DataFrame({
        "stock_code": ["600000.SH", "600000.SH"],
        "trade_date": pd.to_datetime(["2024-02-29", "2024-05-31"]),
        "close": [10.0, 1.0],
    })
    return events, weights, prices


def test_announced_blocks_future_announcement_but_upper_bound_includes_it():
    events, weights, prices = _pit_inputs(ann_date="2024-04-15")
    kwargs = dict(
        valuation_date=pd.Timestamp("2024-03-01"), expiry_date=pd.Timestamp("2024-09-20"),
        index_code="000300", index_level=4000.0, dividend_events=events,
        weight_snapshots=weights, price_history=prices,
    )
    assert point_in_time_div_points(**kwargs, variant="announced_dividend") == 0.0
    assert point_in_time_div_points(**kwargs, variant="realized_dividend_upper_bound") == pytest.approx(8.0)


def test_upper_bound_does_not_require_announcement_column():
    events, weights, prices = _pit_inputs()
    events = events.drop(columns="ann_date")
    kwargs = dict(
        valuation_date=pd.Timestamp("2024-03-01"), expiry_date=pd.Timestamp("2024-09-20"),
        index_code="000300", index_level=4000.0, dividend_events=events,
        weight_snapshots=weights, price_history=prices,
    )
    assert point_in_time_div_points(**kwargs, variant="announced_dividend") == 0.0
    assert point_in_time_div_points(
        **kwargs, variant="realized_dividend_upper_bound"
    ) == pytest.approx(8.0)


def test_asof_weight_and_price_never_use_future_snapshot():
    events, weights, prices = _pit_inputs()
    points = point_in_time_div_points(
        pd.Timestamp("2024-04-01"), pd.Timestamp("2024-09-20"), "000300", 4000.0,
        events, weights, prices, "announced_dividend",
    )
    # 使用 2% 权重和 10 元价格；若泄漏 5 月快照会得到极端错误结果。
    assert points == pytest.approx(8.0)


def test_three_variants_keep_identical_panel_rows_and_raw_is_zero():
    events, weights, prices = _pit_inputs()
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-04-01", "2024-04-02"]),
        "contract": ["IF2409", "IF2409"], "index_code": ["000300", "000300"],
        "month_tag": ["季月", "季月"], "F": [3980.0, 3985.0], "S": [4000.0, 4005.0],
        "days_to_expiry": [120, 119], "calendar_days_to_expiry": [172, 171],
    })
    outputs = {
        variant: dividend_adjust_panel_point_in_time(panel, events, weights, prices, variant)
        for variant in ("raw_basis", "announced_dividend", "realized_dividend_upper_bound")
    }
    assert all(len(out) == len(panel) for out in outputs.values())
    assert all(out[["date", "contract"]].equals(panel[["date", "contract"]]) for out in outputs.values())
    assert outputs["raw_basis"]["div_points"].eq(0).all()
    assert not outputs["announced_dividend"]["contains_lookahead"].any()
    assert outputs["realized_dividend_upper_bound"]["contains_lookahead"].all()
