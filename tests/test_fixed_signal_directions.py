from __future__ import annotations

from dataclasses import replace

import pandas as pd

from basis_model import config, strategy
from basis_model.compare_signal_directions import evaluate_fixed_directions


def test_causal_threshold_does_not_change_when_future_data_changes():
    dates = pd.bdate_range("2024-01-01", periods=8)
    contract = pd.Series(["CONT"] * 8)
    original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    changed = original.copy()
    changed.iloc[5:] = [1000.0, -1000.0, 9999.0]
    first = strategy.causal_basis_thresholds(original, contract, pd.Series(dates), 4, 0.1, 3)
    second = strategy.causal_basis_thresholds(changed, contract, pd.Series(dates), 4, 0.1, 3)
    pd.testing.assert_frame_equal(first.iloc[:6], second.iloc[:6])
    # 第4日阈值只能来自前三日；当天的4不在阈值样本中。
    assert first.loc[3, "upper_threshold"] <= 3.0


def test_two_sided_signal_has_correct_extreme_directions():
    basis = pd.Series([0.0, 0.0, 0.0, -10.0, 10.0])
    signal = strategy.fixed_direction_signal(
        basis, pd.Series(["CONT"] * 5), pd.Series(pd.bdate_range("2024-01-01", periods=5)),
        window=4, pct=0.1, min_periods=3, rule="two_sided",
    )
    assert signal.iloc[3] == 1
    assert signal.iloc[4] == -1


def test_all_strategies_share_dates_and_benchmarks_are_correct():
    dates = pd.bdate_range("2024-01-01", periods=30)
    panel = pd.DataFrame({
        "date": dates, "contract": ["IF2406"] * 30, "index_code": ["000300"] * 30,
        "month_tag": ["当月"] * 30, "S": [4000.0] * 30,
        "adj_basis": pd.Series(range(30), dtype=float).sub(15.0),
        "f1_ann_basis": pd.Series(range(30), dtype=float).sub(15.0).div(100.0),
    })
    cfg = replace(config.DEFAULT_CONFIG, signal_direction_window=10, signal_direction_min_periods=5)
    result, signals = evaluate_fixed_directions(panel, "000300", "raw_basis", cfg)
    assert set(result["strategy"]) == {
        "fixed_momentum_long", "fixed_premium_long", "fixed_two_sided", "buy_and_hold", "flat"
    }
    assert result["n_dates"].nunique() == 1
    assert signals["buy_and_hold"].eq(1).all()
    assert signals["flat"].eq(0).all()
    flat = result[result["strategy"] == "flat"].iloc[0]
    assert flat["annualized_return"] == 0.0
    assert flat["annualized_vol"] == 0.0
