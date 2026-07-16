"""strategy 回测方法的单测：验证仓位序列法(backtest_position_series)修复了
朴素重叠窗口回测(backtest_naive_overlapping)的双重计数/虚高问题。
"""

import numpy as np
import pandas as pd
import pytest

from basis_model import factors as factors_mod
from basis_model import strategy as strategy_mod


def _synthetic_panel(seed: int = 7, n: int = 250, noise_std: float = 1.0):
    rng = np.random.default_rng(seed)
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=n))
    contract = pd.Series(["IF2409"] * n)
    S = pd.Series([4000.0] * n)
    noise = rng.normal(0, noise_std, n).cumsum()
    adj_basis = pd.Series(-20.0 + noise)
    return rng, dates, contract, S, adj_basis


def test_position_series_matches_buy_and_hold_for_constant_signal():
    """信号恒为 +1 时，仓位序列法的净值应等于标的的 buy-and-hold（在容差内）。"""
    _, dates, contract, S, adj_basis = _synthetic_panel()
    ret_1d = factors_mod.basis_daily_return(adj_basis, S, contract, dates)

    n = len(dates)
    signal = pd.Series([1] * n)
    result = strategy_mod.backtest_position_series(dates, signal, ret_1d, N=8)

    buy_and_hold_equity = (1.0 + ret_1d.dropna()).cumprod()

    strategy_final = result["equity_curve"]["equity"].iloc[-1]
    buy_and_hold_final = buy_and_hold_equity.iloc[-1]

    assert strategy_final == pytest.approx(buy_and_hold_final, rel=1e-9)


def test_position_series_avoids_overlapping_window_inflation():
    """同一组信号下，仓位序列法的年化收益量级应远低于朴素重叠窗口版本
    （验证重叠 N 日窗口的双重计数问题不再出现）。
    """
    rng, dates, contract, S, adj_basis = _synthetic_panel()
    N = 8

    ret_1d = factors_mod.basis_daily_return(adj_basis, S, contract, dates)
    y_overlapping = factors_mod.basis_convergence_rate(adj_basis, S, contract, dates, N=N)

    n = len(dates)
    signal = pd.Series(rng.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2]))

    position_result = strategy_mod.backtest_position_series(dates, signal, ret_1d, N=N)
    naive_result = strategy_mod.backtest_naive_overlapping(dates, signal, y_overlapping)

    position_magnitude = abs(position_result["annualized_return"])
    naive_magnitude = abs(naive_result["annualized_return"])

    # 朴素重叠窗口版本的年化收益量级应远大于仓位序列法（实测约 1800+ 倍），
    # 这里用一个远小于实测值、但足以证明"没有再次出现双重计数放大"的阈值。
    assert naive_magnitude > 50 * position_magnitude
    assert naive_result["note"] == "illustrative only — 收益被重叠窗口放大，不可用于横向对比"


def test_position_series_rejects_multiple_contract_rows_per_day():
    dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-02"]))
    with pytest.raises(ValueError, match="每个交易日只有一条"):
        strategy_mod.backtest_position_series(
            dates=dates,
            signal=pd.Series([1, -1]),
            ret_1d=pd.Series([0.01, -0.01]),
            N=8,
        )
