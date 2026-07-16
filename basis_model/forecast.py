"""forecast.py —— 三层预测：分红 -> 回归外推 -> 资金行为叠加。

方法论性质：本模块是三种性质的叠加，务必分清楚各自的权重与可信度：
    1. 分红点位            : 结构建模（dividend.py 已算好，这里只是把它作为基准项接入）
    2. 回归外推 (硬, hard)  : 统计建模 —— regression.py walk-forward 估出的系数，
                              喂入"今天已知"的 f1/f3，做样本外一步预测。
    3. 资金行为叠加 (软, soft) : 定性判断 —— behavioral_adjust() 目前是占位桩函数，
                              未来可替换为雪球/Alpha 资金流显式模型（升级钩子）。

【防前视纪律（纪律 2，本模块是最终防线）】
    forecast_basis() 的参数签名里**没有**、也【不允许】出现 f2_future_return 或
    y_convergence_rate —— 无论是直接参数还是通过 **kwargs 偷渡进来。函数一开始
    就会对所有传入的额外上下文调用 factors.assert_no_lookahead() 做防御性检查。
    注意：expected_return 是用户显式指定的"情景假设"（sensitivity_table 的横轴），
    不是真实发生过的未来收益率，属于合法的假设性输入 —— 但即便如此，它也只允许
    进入 from_qualitative（软）通路，绝不允许进入 from_regression（硬）通路，
    这正是"硬软分离"纪律的体现：统计模型的外推结果不应该因为一个假设性情景就
    改变，改变的只能是定性判断部分。

【硬软分离纪律（纪律 3）】
    forecast_basis() 返回的 dict 必须同时包含 from_regression（硬）与
    from_qualitative（软）两个独立子结构，调用方可以选择只信硬的、只看软的、
    或者自行加权，但两者不能被本模块悄悄合并成一个数字后再返回。
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from . import factors as factors_mod
from . import regression as regression_mod


def behavioral_adjust(snowball_state: Optional[dict], alpha_trend: float) -> float:
    """【定性判断】资金行为叠加，占位桩函数。

    当前实现是极简线性启发式，**不是**真实的资金流建模，只是为了让 forecast_basis
    的 from_qualitative 通路能跑通、有地方接入未来的真实模型。

    TODO 升级钩子：
        - 雪球产品的敲入/敲出临近度 -> 对冲盘 Delta 变化 -> 基差冲击的显式建模。
        - Alpha 策略资金流（多头 Alpha 增减仓）对基差的趋势性影响建模。

    Args:
        snowball_state: 占位字典，预留字段如
            {"knock_in_proximity": 0~1 越接近敲入线越大, "notional": 名义本金}。
        alpha_trend: 占位标量，Alpha 资金趋势强度（正=趋势性加仓，负=趋势性减仓）。

    Returns:
        定性调整点位（可正可负），仅供情景讨论参考。
    """
    snowball_state = snowball_state or {}
    knock_in_pressure = float(snowball_state.get("knock_in_proximity", 0.0))
    hedging_flow_points = -knock_in_pressure * 5.0
    trend_points = float(alpha_trend) * 2.0
    return hedging_flow_points + trend_points


def forecast_basis(
    date: pd.Timestamp,
    contract: str,
    index_code: str,
    S: float,
    current_adj_basis: float,
    f1_ann_basis: float,
    f3_term_structure: float,
    coef_row: pd.Series,
    N: int,
    expected_return: float = 0.0,
    month_tag: Optional[str] = None,
    month_dummy_values: Optional[dict[str, float]] = None,
    snowball_state: Optional[dict] = None,
    alpha_trend: float = 0.0,
    extra_context: Optional[dict] = None,
) -> dict:
    """三层预测的最终出口：分红(已在 current_adj_basis 中体现) -> 回归外推(硬) -> 资金行为(软)。

    Args:
        date, contract, index_code: 预测标的定位信息。
        S: 当前现货点位。
        current_adj_basis: 当前分红调整基差（来自 dividend.dividend_adjusted_basis）。
        f1_ann_basis, f3_term_structure: 当天已知的因子值（factors.py 产出）。
        coef_row: regression.rolling_ols() 某一期的系数（walk-forward，样本外）。
        N: 预测 horizon（交易日数），须与训练 y 用的 N 一致。
        expected_return: 情景假设的未来 N 日现货收益率；通过历史
            估计的「收益率 × 月份」交互项进入 from_regression。它是条件情景，
            不是在预测时偷看到的已实现未来数据。
        month_dummy_values: 合约月份哑元取值（与训练时的 D_month_* 对齐）。
        snowball_state, alpha_trend: behavioral_adjust() 的占位输入。
        extra_context: 调用方可传入的额外上下文，仅用于防前视自检
            （不会被用于任何计算），若混入 f2_future_return / y_convergence_rate
            会直接抛出 AssertionError。

    Returns:
        dict: {
            "date", "contract", "index_code",
            "direction": str,                # 综合方向描述（以硬预测为准）
            "points": float,                 # 主输出点位 = from_regression 的硬预测
            "from_regression": {...},        # 硬：统计回归外推
            "from_qualitative": {...},       # 软：定性资金行为叠加（含情景假设）
        }
    """
    # 防前视自检：绝不允许训练专用列混进预测输入
    if extra_context:
        factors_mod.assert_no_lookahead(extra_context)

    active_month_tag = month_tag
    if active_month_tag is None and month_dummy_values:
        active = [k.removeprefix("D_month_") for k, v in month_dummy_values.items() if v == 1]
        active_month_tag = active[0] if active else "当月"
    y_hat = regression_mod.predict_convergence_rate(
        coef_row,
        f1_ann_basis,
        f3_term_structure,
        month_dummy_values,
        expected_return=expected_return,
        month_tag=active_month_tag,
    )
    predicted_basis_change = y_hat * S * N / 360.0
    predicted_future_adj_basis = current_adj_basis + predicted_basis_change

    from_regression = {
        "predicted_convergence_rate": y_hat,
        "predicted_basis_change_points": predicted_basis_change,
        "predicted_future_adj_basis": predicted_future_adj_basis,
        "note": "统计外推（walk-forward OLS 样本外预测），硬输出。",
    }

    qualitative_points = behavioral_adjust(snowball_state, alpha_trend)
    scenario_sensitivity_points = 0.0
    from_qualitative = {
        "expected_return_scenario": expected_return,
        "behavioral_adjustment_points": qualitative_points,
        "scenario_sensitivity_points": scenario_sensitivity_points,
        "total_qualitative_points": qualitative_points + scenario_sensitivity_points,
        "note": "定性资金行为调整（占位模型），软输出；收益率情景已进入统计回归通路。",
    }

    eps = 1e-9
    if predicted_basis_change > eps:
        direction = "basis上行（贴水收窄/升水走扩）"
    elif predicted_basis_change < -eps:
        direction = "basis下行（贴水走深/升水收窄）"
    else:
        direction = "基本持平"

    return {
        "date": date,
        "contract": contract,
        "index_code": index_code,
        "direction": direction,
        "points": predicted_basis_change,
        "from_regression": from_regression,
        "from_qualitative": from_qualitative,
    }


def sensitivity_table(
    date: pd.Timestamp,
    contract: str,
    index_code: str,
    S: float,
    current_adj_basis: float,
    f1_ann_basis: float,
    f3_term_structure: float,
    coef_row: pd.Series,
    N: int,
    expected_returns: Sequence[float] = (-0.04, -0.02, 0.0, 0.02, 0.04),
    month_tag: Optional[str] = None,
    month_dummy_values: Optional[dict[str, float]] = None,
    snowball_state: Optional[dict] = None,
    alpha_trend: float = 0.0,
) -> pd.DataFrame:
    """情景式主输出：对一组假设的未来现货收益率情景，展示硬/软预测结果。

    expected_return 是用户给定的条件情景，会通过历史估计的
    X_return_<month_tag> 系数改变 from_regression；它不是前视数据。
    """
    rows = []
    for er in expected_returns:
        result = forecast_basis(
            date=date,
            contract=contract,
            index_code=index_code,
            S=S,
            current_adj_basis=current_adj_basis,
            f1_ann_basis=f1_ann_basis,
            f3_term_structure=f3_term_structure,
            coef_row=coef_row,
            N=N,
            expected_return=er,
            month_tag=month_tag,
            month_dummy_values=month_dummy_values,
            snowball_state=snowball_state,
            alpha_trend=alpha_trend,
        )
        rows.append(
            {
                "expected_return_scenario": er,
                "direction": result["direction"],
                "from_regression_points": result["from_regression"]["predicted_basis_change_points"],
                "from_qualitative_points": result["from_qualitative"]["total_qualitative_points"],
                "combined_points": (
                    result["from_regression"]["predicted_basis_change_points"]
                    + result["from_qualitative"]["total_qualitative_points"]
                ),
            }
        )
    return pd.DataFrame(rows)
