"""dividend.py —— 分红点位预测 + 分红调整基差。

方法论性质：**结构建模**（不是统计回归，而是把已知的公司分红行为，按照
指数编制规则折算成"未来会打入现货指数点位的确定性缺口"，再用它去修正基差）。

核心恒等式：
    名义基差       = F - S
    分红调整基差    = (F - S) + div_points
    其中 div_points 是"预测日到合约到期日之间，将要发生的除权除息"对指数点位的
    影响之和 —— 因为现货指数在除息日会因为成分股除权而"无理由"下跌，这部分
    下跌不是基差真实收敛，必须先从名义基差里剔除，否则会污染下游的因子/回归。

分红点位折算（近似公式，研究工具意义上的"够用"精度）：
    点位贡献_i = 股息率_i × 权重_i(%) / 100 × 指数点位
    股息率_i   = 每股股利_i / 股价_i   （data.py 已在 load_dividends 中算好）

只累加"预测日之后、合约到期前"的除息事件，避免未来分红污染当期基差、也避免
把已经除权过的历史分红重复计入。

【assume_zero_dividend 开关】
    原型阶段（分红模块尚未打磨/成分股权重数据不全）默认打开 assume_zero_dividend=True，
    此时 div_points 恒为 0，adj_basis 退化为名义基差 (F-S)，让下游因子/回归/策略/
    预测四层可以先跑通、先验证统计规律，再在正式研究阶段关掉这个开关接入真实分红。

顺序刚性（纪律 1）：本模块必须在 factors.py 之前调用 —— 先剔分红，才能造因子。
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from . import data as data_mod

#: 经验上的分红高峰月份（5-7月为除息高峰）
DIVIDEND_SEASON_MONTHS: tuple[int, ...] = (5, 6, 7)
DividendVariant = Literal["raw_basis", "announced_dividend", "realized_dividend_upper_bound"]


def is_dividend_season(ts: pd.Timestamp) -> bool:
    """是否处于除息高峰期（5-7月）。"""
    return pd.Timestamp(ts).month in DIVIDEND_SEASON_MONTHS


def div_points(
    valuation_date: pd.Timestamp,
    expiry_date,
    index_code: str,
    dividend_events: pd.DataFrame,
    index_level: float,
    assume_zero_dividend: bool = True,
) -> float:
    """计算 [valuation_date, expiry_date] 区间内、某指数的预期分红点位。

    只累加"预测日之后（不含当日）、合约到期前（含到期日）"的除息事件：
        valuation_date < ex_date <= expiry_date

    Args:
        valuation_date: 预测/估值基准日。
        expiry_date: 合约到期日（datetime.date 或 pd.Timestamp）。
        index_code: 指数代码。
        dividend_events: data.py.load_dividends 输出的分红面板
            (stock_code, index_code, ex_date, dividend_per_share, weight_pct, dividend_yield)。
        index_level: 折算用的指数点位（一般用现货 S）。
        assume_zero_dividend: 原型阶段开关，True 时直接返回 0（默认打开）。

    Returns:
        预期分红点位（正数，代表指数会因除息"损失"的点位）。
    """
    if assume_zero_dividend:
        return 0.0
    if dividend_events is None or dividend_events.empty:
        return 0.0

    valuation_ts = pd.Timestamp(valuation_date)
    expiry_ts = pd.Timestamp(expiry_date)

    mask = (
        (dividend_events["index_code"] == index_code)
        & (dividend_events["ex_date"] > valuation_ts)
        & (dividend_events["ex_date"] <= expiry_ts)
    )
    subset = dividend_events.loc[mask]
    if subset.empty:
        return 0.0

    contribution = subset["dividend_yield"] * subset["weight_pct"] / 100.0 * index_level
    return float(contribution.sum())


def dividend_adjusted_basis(F: pd.Series | float, S: pd.Series | float, div_points_: pd.Series | float):
    """分红调整基差 = (F - S) + div_points。

    这是本模块唯一的核心恒等式，向量化实现，标量/Series 均可。
    """
    return (F - S) + div_points_


def _asof_snapshot(
    snapshots: pd.DataFrame,
    stock_code: str,
    valuation_date: pd.Timestamp,
    date_col: str,
) -> Optional[pd.Series]:
    """取同一股票 ``date_col <= valuation_date`` 的最近快照，禁止未来泄漏。"""
    eligible = snapshots[
        snapshots["stock_code"].astype(str).eq(str(stock_code))
        & (pd.to_datetime(snapshots[date_col]) <= valuation_date)
    ]
    if eligible.empty:
        return None
    return eligible.loc[pd.to_datetime(eligible[date_col]).idxmax()]


def point_in_time_div_points(
    valuation_date: pd.Timestamp,
    expiry_date,
    index_code: str,
    index_level: float,
    dividend_events: pd.DataFrame,
    weight_snapshots: pd.DataFrame,
    price_history: pd.DataFrame,
    variant: DividendVariant,
) -> float:
    """计算 raw/announced/upper-bound 三口径的分红点位。

    announced 只允许 ``ann_date <= t``；upper_bound 故意忽略公告日，含前视，
    仅作理论上界。权重和股价均严格取 ``trade_date <= t`` 的最近快照。
    """
    if variant == "raw_basis":
        return 0.0
    required_events = {"stock_code", "index_code", "ex_date", "cash_div"}
    if dividend_events.empty or not required_events.issubset(dividend_events.columns):
        return 0.0
    t, expiry = pd.Timestamp(valuation_date), pd.Timestamp(expiry_date)
    events = dividend_events.copy()
    if "ann_date" in events:
        events["ann_date"] = pd.to_datetime(events["ann_date"], errors="coerce")
    else:
        events["ann_date"] = pd.NaT
    events["ex_date"] = pd.to_datetime(events["ex_date"], errors="coerce")
    mask = (
        events["index_code"].astype(str).eq(str(index_code))
        & (events["ex_date"] > t)
        & (events["ex_date"] <= expiry)
    )
    if variant == "announced_dividend":
        mask &= events["ann_date"].notna() & (events["ann_date"] <= t)
    elif variant != "realized_dividend_upper_bound":
        raise ValueError(f"未知分红口径: {variant}")
    events = events.loc[mask].copy()
    if events.empty:
        return 0.0
    # 同一年度分配方案可能多次公告更新；每个估值时点只保留当时最新版本。
    group_cols = ["stock_code", "ex_date"]
    if "end_date" in events.columns:
        group_cols.append("end_date")
    events = events.sort_values("ann_date", na_position="first").groupby(group_cols, dropna=False).tail(1)

    points = 0.0
    for event in events.itertuples(index=False):
        weight = _asof_snapshot(weight_snapshots, event.stock_code, t, "trade_date")
        price = _asof_snapshot(price_history, event.stock_code, t, "trade_date")
        if weight is None or price is None:
            continue
        weight_pct = pd.to_numeric(weight.get("weight_pct"), errors="coerce")
        close = pd.to_numeric(price.get("close"), errors="coerce")
        cash_div = pd.to_numeric(event.cash_div, errors="coerce")
        if pd.isna(weight_pct) or pd.isna(close) or pd.isna(cash_div) or close <= 0:
            continue
        points += float(cash_div / close * weight_pct / 100.0 * index_level)
    return points


def dividend_adjust_panel_point_in_time(
    panel: pd.DataFrame,
    dividend_events: pd.DataFrame,
    weight_snapshots: pd.DataFrame,
    price_history: pd.DataFrame,
    variant: DividendVariant,
) -> pd.DataFrame:
    """在完全相同的输入面板上生成指定分红口径，保持行数与行序不变。"""
    out = panel.copy()
    if variant == "raw_basis":
        out["div_points"] = 0.0
    else:
        out["div_points"] = [
            point_in_time_div_points(
                row.date, data_mod.cffex_expiry(row.contract), row.index_code, row.S,
                dividend_events, weight_snapshots, price_history, variant,
            )
            for row in out.itertuples(index=False)
        ]
    out["basis_variant"] = variant
    out["contains_lookahead"] = variant == "realized_dividend_upper_bound"
    out["adj_basis"] = dividend_adjusted_basis(out["F"], out["S"], out["div_points"])
    return out


def dividend_adjust_panel(
    panel: pd.DataFrame,
    dividend_events: Optional[pd.DataFrame] = None,
    assume_zero_dividend: bool = True,
) -> pd.DataFrame:
    """对整张面板逐行计算 div_points 与 adj_basis，返回新增两列的面板副本。

    要求 panel 至少含 date, contract, index_code, F, S 列（data.py 输出的 schema）。
    """
    out = panel.copy()
    if assume_zero_dividend or dividend_events is None or dividend_events.empty:
        out["div_points"] = 0.0
    else:
        points = []
        for row in out.itertuples(index=False):
            expiry = data_mod.cffex_expiry(row.contract)
            p = div_points(
                valuation_date=row.date,
                expiry_date=expiry,
                index_code=row.index_code,
                dividend_events=dividend_events,
                index_level=row.S,
                assume_zero_dividend=False,
            )
            points.append(p)
        out["div_points"] = points

    out["adj_basis"] = dividend_adjusted_basis(out["F"], out["S"], out["div_points"])
    return out
