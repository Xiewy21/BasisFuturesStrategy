from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from basis_model import config
from basis_model.walkforward_direction import (
    CANDIDATE_RULES,
    run_walkforward,
    select_direction_from_training,
)


def _panel(periods: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    basis = 25.0 * np.sin(np.arange(periods) / 18.0) + np.arange(periods) * 0.002
    return pd.DataFrame({
        "date": dates,
        "contract": ["IF_CONT"] * periods,
        "index_code": ["000300"] * periods,
        "month_tag": ["当月"] * periods,
        "S": [4000.0] * periods,
        "adj_basis": basis,
        "f1_ann_basis": basis / 4000.0 * 252.0 / 20.0,
    })


def test_selection_only_uses_supplied_purged_training_data():
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=60))
    returns = pd.Series(np.r_[np.repeat(0.001, 40), np.repeat(-0.50, 20)])
    signals = {
        name: pd.Series(1 if name == "fixed_momentum_long" else 0, index=range(60))
        for name in CANDIDATE_RULES
    }
    cfg = replace(config.DEFAULT_CONFIG, walkforward_hac_tstat_min=-1.0, walkforward_sharpe_nw_min=-1.0)
    first = select_direction_from_training(
        dates.iloc[:40], returns.iloc[:40], {k: v.iloc[:40] for k, v in signals.items()}, cfg
    )[0]
    # 任意修改“测试段”收益；选择函数仍只收到相同的 purge 后训练切片。
    returns.iloc[40:] = 99.0
    second = select_direction_from_training(
        dates.iloc[:40], returns.iloc[:40], {k: v.iloc[:40] for k, v in signals.items()}, cfg
    )[0]
    assert first == second == "fixed_momentum_long"


def test_walkforward_log_has_purge_and_embargo_on_both_sides():
    cfg = replace(
        config.DEFAULT_CONFIG,
        signal_direction_window=80,
        signal_direction_min_periods=40,
    )
    _, log, _, _ = run_walkforward(_panel(), "000300", "raw_basis", cfg)
    assert not log.empty
    assert log["purge_n_trading_days"].eq(cfg.n_horizon).all()
    assert log["embargo_n_trading_days"].eq(cfg.n_horizon).all()
    assert (log["n_train_nominal"] - log["n_train_after_purge"]).eq(cfg.n_horizon).all()
    assert (log["n_test_nominal"] - log["n_test_after_embargo"]).eq(cfg.n_horizon).all()
    assert (pd.to_datetime(log["train_end_after_purge"]) < pd.to_datetime(log["test_start"])).all()
    assert (pd.to_datetime(log["oos_start"]) > pd.to_datetime(log["embargo_end"])).all()


def test_failed_training_gate_selects_flat_and_oos_return_is_zero():
    cfg = replace(
        config.DEFAULT_CONFIG,
        walkforward_train_years=1,
        signal_direction_window=60,
        signal_direction_min_periods=20,
        walkforward_hac_tstat_min=999.0,
        walkforward_sharpe_nw_min=999.0,
    )
    result, log, counts, _ = run_walkforward(_panel(450), "000300", "raw_basis", cfg)
    assert log["selected_direction"].eq("flat").all()
    dynamic = result.loc[result["strategy"].eq("walkforward_dynamic")].iloc[0]
    assert dynamic["annualized_return"] == 0.0
    assert dynamic["annualized_vol"] == 0.0
    assert counts.loc[counts["selected_direction"].eq("flat"), "share"].iloc[0] == 1.0
