"""factors.py —— 因子构造层。

方法论性质：**统计建模的前置工程**（因子本身是对数据的确定性变换，
但存在的意义是给 regression.py 的统计回归提供解释变量/被解释变量）。

因子定义：
    f1 = ann_basis                        当前年化基差，均值回复因子
    f2 = future_return(S, N)              未来 N 日现货收益率 —— 【仅训练】禁止进预测
    f3 = ann_basis(季月) - ann_basis(下季月)   期限结构斜率因子
    y  = (adj_basis.shift(-N) - adj_basis) / (S * N/360)   N 日基差平均收敛率（回归目标）

    ann_basis = adj_basis / S * 252 / D    （D = 剩余交易日 days_to_expiry）

【防前视纪律（纪律 2）】
    f2 和 y 都用到了 t+N 期的信息，只允许出现在 regression.py 的训练矩阵里。
    forecast.py 的预测路径绝不能读取这两列 —— 本模块用 TRAINING_ONLY_COLUMNS
    显式标注，并提供 assert_no_lookahead() 供其他模块 / 单测强制检查。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 这两列只允许用于 regression.py 的训练矩阵，forecast.py 绝不能读取
TRAINING_ONLY_COLUMNS: tuple[str, ...] = ("f2_future_return", "y_convergence_rate")

RETURN_INTERACTION_PREFIX = "X_return_"


def assert_no_lookahead(obj) -> None:
    """防前视断言：obj 中不得包含 TRAINING_ONLY_COLUMNS 里的任何列/键。

    obj 可以是 DataFrame（检查 columns）、dict（检查 keys），或任意带
    __contains__ 的容器。forecast.py 在构造预测输入时必须调用本函数自检。
    """
    if isinstance(obj, pd.DataFrame):
        leaked = [c for c in TRAINING_ONLY_COLUMNS if c in obj.columns]
    elif isinstance(obj, dict):
        leaked = [c for c in TRAINING_ONLY_COLUMNS if c in obj]
    else:
        leaked = [c for c in TRAINING_ONLY_COLUMNS if c in obj]
    if leaked:
        raise AssertionError(
            f"防前视违规：预测路径中检测到训练专用列/字段 {leaked}，"
            "future_return / y(收敛率) 严禁进入预测路径。"
        )


def ann_basis(adj_basis: pd.Series, S: pd.Series, D: pd.Series) -> pd.Series:
    """年化基差 f1 = adj_basis / S * 252 / D。D<=0 的行返回 NaN。"""
    D_safe = D.where(D > 0, other=np.nan)
    return adj_basis / S * 252.0 / D_safe


def future_return(S: pd.Series, contract: pd.Series, date: pd.Series, N: int) -> pd.Series:
    """未来 N 日现货收益率 f2 = (S_{t+N} - S_t) / S_t，按 contract 分组、date 排序计算。

    【仅训练】：本函数产出的列必须打上 f2_future_return 名字，只能进入
    regression.py 的训练矩阵，forecast.py 不得调用本函数。
    """
    orig_index = S.index
    df = pd.DataFrame(
        {"S": S.values, "contract": contract.values, "date": date.values}, index=orig_index
    )
    df = df.sort_values(["contract", "date"])
    df["f2_future_return"] = df.groupby("contract")["S"].transform(
        lambda s: (s.shift(-N) - s) / s
    )
    return df["f2_future_return"].reindex(orig_index)


def basis_convergence_rate(
    adj_basis: pd.Series, S: pd.Series, contract: pd.Series, date: pd.Series, N: int
) -> pd.Series:
    """回归目标 y = (adj_basis_{t+N} - adj_basis_t) / (S_t * N/360)，按 contract 分组、date 排序。"""
    orig_index = adj_basis.index
    df = pd.DataFrame(
        {
            "adj_basis": adj_basis.values,
            "S": S.values,
            "contract": contract.values,
            "date": date.values,
        },
        index=orig_index,
    )
    df = df.sort_values(["contract", "date"])

    df["adj_basis_fwd"] = df.groupby("contract")["adj_basis"].transform(lambda s: s.shift(-N))
    df["y_convergence_rate"] = (df["adj_basis_fwd"] - df["adj_basis"]) / (df["S"] * N / 360.0)
    return df["y_convergence_rate"].reindex(orig_index)


def basis_daily_return(
    adj_basis: pd.Series, S: pd.Series, contract: pd.Series, date: pd.Series
) -> pd.Series:
    """单日基差收益率（非年化）：ret_1d = (adj_basis_{t+1} - adj_basis_t) / S_t。

    与 basis_convergence_rate 的区别很关键，不要混用：
        - basis_convergence_rate(..., N) 除以 (S*N/360) 是为了把不同 N 的窗口
          折算成可比的"年化收敛速度"，是喂给 regression.py 的因子/目标，
          数值上被放大了约 360/N 倍，不是真实的单期收益率。
        - basis_daily_return 只是最朴素的"今天基差变动了多少个点、相当于
          百分之几"的单日已实现收益率，不做任何年化处理。

    这是 strategy.backtest_position_series() 里 ret_1d 参数唯一正确的口径 ——
    如果误用 basis_convergence_rate(..., N=1) 当作 ret_1d，会因为年化换算的
    360 倍尺度重新引入一次虚高（换了一种方式，但后果和重叠窗口双重计数一样坏）。
    """
    return basis_convergence_rate(adj_basis, S, contract, date, N=1) / 360.0


def term_structure_factor(panel: pd.DataFrame, f1: pd.Series) -> pd.Series:
    """期限结构斜率因子 f3 = ann_basis(季月) - ann_basis(下季月)。

    对每个 (date, index_code)，用当日"季月"合约与"下季月"合约的年化基差之差
    作为市场级期限结构斜率信号，再广播到该 (date, index_code) 下所有合约的行上。
    该因子刻画的是"近端-远端"贴水/升水的陡峭程度，而非单一合约自身的基差水平。

    注：sanity check 参考（某券商 IC, N=8）显示该因子回归系数不显著，属于弱因子，
    保留是为了统计上的完整性与后续升级（如联合期限结构模型）。
    """
    tmp = panel.copy()
    tmp["f1"] = f1.values
    pivot = tmp.pivot_table(
        index=["date", "index_code"], columns="month_tag", values="f1", aggfunc="first"
    )
    if "季月" not in pivot.columns:
        pivot["季月"] = np.nan
    if "下季月" not in pivot.columns:
        pivot["下季月"] = np.nan
    slope = (pivot["季月"] - pivot["下季月"]).rename("f3_term_structure")

    merged = tmp.merge(slope.reset_index(), on=["date", "index_code"], how="left")
    return merged["f3_term_structure"]


def month_dummies(month_tag: pd.Series) -> pd.DataFrame:
    """合约月份虚拟变量 D_month，drop_first 避免与常数项共线。"""
    dummies = pd.get_dummies(month_tag, prefix="D_month", drop_first=True)
    return dummies.astype(float)


def return_month_interactions(future_ret: pd.Series, month_tag: pd.Series) -> pd.DataFrame:
    """构造报告口径的「未来 N 日收益率 × 合约月份」交互项。

    四个月份类型全部保留；回归中不再单独加入 future_return，
    因此不会与这四列的和产生完全共线。这些列只用于历史训练；
    预测时使用的是用户明确给定的 expected_return 情景，不是已实现的未来值。
    """
    tags = ("当月", "次月", "季月", "下季月")
    return pd.DataFrame(
        {
            f"{RETURN_INTERACTION_PREFIX}{tag}": future_ret * (month_tag == tag).astype(float)
            for tag in tags
        },
        index=future_ret.index,
    )


def build_factor_panel(panel: pd.DataFrame, N: int) -> pd.DataFrame:
    """顺序刚性入口：拿到 dividend.py 处理过的面板（含 adj_basis），产出完整因子面板。

    输出列：原始面板列 + f1_ann_basis, f2_future_return(仅训练), f3_term_structure,
    y_convergence_rate(仅训练), D_month_* 虚拟变量。
    """
    required = {"date", "contract", "index_code", "month_tag", "F", "S", "days_to_expiry", "adj_basis"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            f"build_factor_panel 需要 dividend.py 处理过的面板（含 adj_basis 等列），缺少: {missing}"
        )

    out = panel.copy().reset_index(drop=True)
    out["f1_ann_basis"] = ann_basis(out["adj_basis"], out["S"], out["days_to_expiry"])
    out["f2_future_return"] = future_return(out["S"], out["contract"], out["date"], N)
    out["f3_term_structure"] = term_structure_factor(out, out["f1_ann_basis"])
    out["y_convergence_rate"] = basis_convergence_rate(
        out["adj_basis"], out["S"], out["contract"], out["date"], N
    )
    dummies = month_dummies(out["month_tag"])
    interactions = return_month_interactions(out["f2_future_return"], out["month_tag"])
    out = pd.concat([out, dummies, interactions], axis=1)
    return out
