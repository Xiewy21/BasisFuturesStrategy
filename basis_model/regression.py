"""regression.py —— 多因子回归，walk-forward 滚动估参。

方法论性质：**统计建模**。

模型设定：
    y_convergence_rate ~ const + f1_ann_basis + f3_term_structure + D_month_*

纪律：
    - 系数一律 walk-forward 滚动估计，**禁止全样本一次性拟合**。rolling_ols()
      在每个截止日，只用"当天为止、且 y 已经实现（非 NaN）"的历史窗口数据拟合，
      产出该截止日"当时可获得"的系数，供 forecast.py 做样本外外推使用。
    - f2_future_return 是训练专用列，默认以 X_return_<month_tag>
      交互项进入历史回归。forecast.py 不读取该列，而是读取用户
      给定的 expected_return 情景，因此不构成前视。

sanity check 参考系数（某券商 IC, N=8，仅供数量级校验，不是断言标准）：
    a1(当前年化基差) ≈ -1.5082
    季月收益率系数   ≈ 2.0447
    期限结构因子(N=8) 不显著
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

#: sanity check 参考值（某券商 IC, N=8），仅用于量级校验，不作为强断言标准
SANITY_REFERENCE_COEFFICIENTS: dict[str, float] = {
    "index_code": "000905",
    "N": 8,
    "f1_ann_basis": -1.5082,
    "季月收益率(f2)": 2.0447,
    "f3_term_structure_significant": False,
}

DEFAULT_FEATURE_COLS: tuple[str, ...] = ("f1_ann_basis", "f3_term_structure")
TARGET_COL: str = "y_convergence_rate"


def _return_interaction_cols(factor_panel: pd.DataFrame) -> list[str]:
    return [c for c in factor_panel.columns if c.startswith("X_return_")]


def _month_dummy_cols(factor_panel: pd.DataFrame) -> list[str]:
    return [c for c in factor_panel.columns if c.startswith("D_month_")]


def build_design_matrix(
    factor_panel: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    include_month_dummies: bool = True,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """从因子面板中抽取训练用的 (X, y)，仅保留 y 已实现（非 NaN）的行。"""
    cols = (
        list(feature_cols)
        if feature_cols is not None
        else list(DEFAULT_FEATURE_COLS) + _return_interaction_cols(factor_panel)
    )
    if include_month_dummies:
        cols = cols + _month_dummy_cols(factor_panel)

    subset = factor_panel.dropna(subset=cols + [target_col])
    X = sm.add_constant(subset[cols], has_constant="add")
    y = subset[target_col]
    return X, y


def rolling_ols(
    factor_panel: pd.DataFrame,
    window: int,
    min_obs: int = 30,
    step: int = 1,
    feature_cols: Optional[Sequence[str]] = None,
    include_month_dummies: bool = True,
    target_col: str = TARGET_COL,
    hac_lags: Optional[int] = None,
) -> pd.DataFrame:
    """walk-forward 滚动 OLS，返回按截止日索引的系数时间序列。

    在每个截止日 cutoff，只使用"date <= cutoff 且 y 已实现"的最近 window 个
    交易日数据拟合 OLS；产出的系数代表"站在 cutoff 这一天、只用当时已知信息"
    能估出的模型参数，可直接喂给 forecast.py 做样本外预测，不构成前视。

    Args:
        factor_panel: factors.build_factor_panel 的输出。
        window: 滚动窗口的交易日数量（按 unique date 计）。
        min_obs: 窗口内最少观测数，不足则跳过该截止日（不产出系数）。
        step: 每隔多少个交易日重新估计一次（默认每日）。
        feature_cols: 自变量列名，默认 f1_ann_basis + f3_term_structure。
        include_month_dummies: 是否加入 D_month_* 虚拟变量。
        target_col: 因变量列名，默认 y_convergence_rate。
        hac_lags: 若给定（一般取 N），用 Newey-West（HAC）协方差矩阵拟合，
            额外产出 tstat_<col>/pvalue_<col> 列，避免重叠 N 日目标变量自身带来
            的自相关让显著性被经典 OLS 标准误高估。系数点估计不受协方差类型影响，
            只有标准误/t值/p值会变。单个窗口内 HAC 拟合失败（如样本量不足以支撑
            该滞后阶）时该窗口退化为经典 OLS，不中断整条时间序列。

    Returns:
        DataFrame，index=date，columns=[n_obs, const, f1_ann_basis, f3_term_structure,
        D_month_*, r_squared]，hac_lags 给定时额外含 tstat_*/pvalue_* 列。
    """
    cols = (
        list(feature_cols)
        if feature_cols is not None
        else list(DEFAULT_FEATURE_COLS) + _return_interaction_cols(factor_panel)
    )
    if include_month_dummies:
        cols = cols + _month_dummy_cols(factor_panel)

    dates = sorted(factor_panel["date"].unique())
    records = []
    for i in range(0, len(dates), step):
        cutoff = dates[i]
        window_dates = [d for d in dates if d <= cutoff][-window:]
        if len(window_dates) < min(window, min_obs):
            continue
        train = factor_panel[factor_panel["date"].isin(window_dates)]
        train = train.dropna(subset=cols + [target_col])
        if len(train) < min_obs:
            continue

        X = sm.add_constant(train[cols], has_constant="add")
        y = train[target_col]

        model = None
        if hac_lags is not None and len(train) > hac_lags + 2:
            try:
                model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
            except Exception:
                model = None
        if model is None:
            model = sm.OLS(y, X).fit()

        record = {"date": cutoff, "n_obs": len(train), "r_squared": model.rsquared}
        record.update(model.params.to_dict())
        if hac_lags is not None:
            record.update(model.tvalues.add_prefix("tstat_").to_dict())
            record.update(model.pvalues.add_prefix("pvalue_").to_dict())
        records.append(record)

    if not records:
        return pd.DataFrame(columns=["date", "n_obs", "r_squared", "const"] + cols).set_index(
            "date"
        )
    return pd.DataFrame(records).set_index("date")


def predict_convergence_rate(
    coef_row: pd.Series,
    f1_ann_basis: float,
    f3_term_structure: float,
    month_dummy_values: Optional[dict[str, float]] = None,
    expected_return: float = 0.0,
    month_tag: Optional[str] = None,
) -> float:
    """用某一期 walk-forward 系数，对新的 f1/f3（当天已知量）做样本外预测。

    只吃 f1/f3 与月份哑元，绝不接受 f2_future_return 或 y_convergence_rate ——
    这是 forecast.py 防前视纪律得以成立的关键：回归系数本身不含未来信息，
    预测输入也不含未来信息，两边都干净。
    """
    y_hat = coef_row.get("const", 0.0)
    y_hat += coef_row.get("f1_ann_basis", 0.0) * f1_ann_basis
    y_hat += coef_row.get("f3_term_structure", 0.0) * f3_term_structure
    if month_dummy_values:
        for k, v in month_dummy_values.items():
            if k in coef_row.index:
                y_hat += coef_row[k] * v
    if month_tag:
        y_hat += coef_row.get(f"X_return_{month_tag}", 0.0) * expected_return
    return float(y_hat)


# --------------------------------------------------------------------------- #
# 升级钩子（预留接口，不实现）
# --------------------------------------------------------------------------- #

def ou_mean_reversion_params(*args, **kwargs):
    """TODO 升级钩子：用 Ornstein-Uhlenbeck 过程估计基差均值回复速度/长期均值/波动率，
    替代当前的线性 OLS 假设。未实现。"""
    raise NotImplementedError("升级钩子，尚未实现：OU 均值回复过程估参")


def kalman_dynamic_coefficients(*args, **kwargs):
    """TODO 升级钩子：用卡尔曼滤波做时变系数估计，替代固定窗口 walk-forward OLS。未实现。"""
    raise NotImplementedError("升级钩子，尚未实现：卡尔曼动态系数")


def walk_forward_oos_validation(*args, **kwargs):
    """TODO 升级钩子：系统化的 walk-forward 样本外验证框架（滚动切分、
    多窗口/多步长网格搜索、样本外 R2/IC 统计）。未实现。"""
    raise NotImplementedError("升级钩子，尚未实现：walk-forward 样本外验证")
