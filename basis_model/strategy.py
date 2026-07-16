"""strategy.py —— 对冲成本 / 套利收益率 / 择时信号 + 回测。

方法论性质分区（模块内三种性质并存，务必分清楚）：
    - hedge_cost / best_rollover                       : 结构建模（确定性核算）
    - forward_arb / reverse_arb                          : 结构建模（无套利定价关系）
    - timing_signal / map_by_index / backtest_*          : 【规则回测】—— 这是研究工具
      里"用历史数据跑一遍规则、看收益曲线长什么样"的意思，**不是**一个被统计检验
      证明有效的模型，更不能直接当成实盘交易信号。样本内过拟合、成本假设简化、
      无滑点/冲击成本等问题都没有处理，仅供研究阶段快速看一眼规则的历史表现。

回测方法二选一，务必按场景选对：
    - backtest_position_series() 【主推荐】：仓位序列法，每日只结算 1 日收益，
      不会重复计数，是本模块唯一"可交易表述"的回测方法，年化/夏普/回撤等指标
      可以互相横向比较。
    - backtest_naive_overlapping()：保留的旧版重叠窗口回测，**仅作示意
      (illustrative only)**——如果喂给它的收益率是重叠的 N 日窗口（比如
      factors.y_convergence_rate），逐日复利会把同一段行情重复计入多次，
      年化收益率会被显著放大且不可交易，**不能用于横向对比**，只用于快速
      感受"这个方向大致对不对"。
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import regression as regression_mod

# --------------------------------------------------------------------------- #
# 结构建模：对冲成本 / 展期选择
# --------------------------------------------------------------------------- #

def hedge_cost(basis: pd.Series, contract: pd.Series, date: pd.Series, N: int) -> pd.Series:
    """对冲成本 = basis_{t+N} - basis_t，按 contract 分组、date 排序计算。

    含义：用期货空头对冲现货多头的人，在展期/平仓时实际承担的基差变动成本。
    basis 变得更正（升水）对空头对冲者不利，更负（贴水加深）也不利（取决于
    开仓时基差符号），这里只做核算，不做方向判断。
    """
    orig_index = basis.index
    df = pd.DataFrame(
        {"basis": basis.values, "contract": contract.values, "date": date.values},
        index=orig_index,
    )
    df = df.sort_values(["contract", "date"])
    df["hedge_cost"] = df.groupby("contract")["basis"].transform(lambda s: s.shift(-N) - s)
    return df["hedge_cost"].reindex(orig_index)


def best_rollover(
    candidates: pd.DataFrame,
    predicted_col: str = "y_hat_convergence_rate",
) -> Optional[str]:
    """从同一天的候选合约中，选出"预测收敛最慢"的合约用于展期。

    "收敛最慢" = 预测的基差收敛率(|y_hat|)绝对值最小 —— 对于需要滚动展期的
    对冲盘而言，基差变动越小，展期时侵蚀掉的对冲成本就越小。

    Args:
        candidates: 同一天、同一标的指数下多个合约的候选集，需含 contract 列
            与 predicted_col 列（一般来自 regression.predict_convergence_rate）。
        predicted_col: 预测收敛率所在列名。

    Returns:
        推荐展期的合约代码；candidates 为空时返回 None。
    """
    if candidates.empty:
        return None
    valid = candidates.dropna(subset=[predicted_col])
    if valid.empty:
        return None
    idx = valid[predicted_col].abs().idxmin()
    return str(valid.loc[idx, "contract"])


def hedge_cost_ladder(
    factor_panel: pd.DataFrame,
    coefs_daily: pd.DataFrame,
    N: int,
) -> pd.DataFrame:
    """用真实换约事件比较四种空头对冲展仓规则。

    符号严格站在“持有期货空头对冲现货多头”的视角：换约时买平旧合约、
    卖开新合约，因此 ``roll_cost_points = F_old - F_new``；正数表示付出成本，
    负数表示取得展仓收益。不能使用 ``F_new-F_old``，否则符号会反转。

    四种规则：
        - 当月：始终用 month_tag=='当月' 的合约。
        - 季月：始终用 month_tag=='季月' 的合约。
        - 最低贴水：每天在当日所有在市合约里，选 |f1_ann_basis| 最小（贴水/升水
          幅度最小）的合约。
        - 回归展仓：每天用 coefs_daily 里当天可得的 walk-forward 系数，对每个
          候选合约算出预测收敛率 y_hat，调用 best_rollover() 选预测收敛最慢的合约。

    四条路径使用相同的逐日评估集合、价格类型和缺失处理，但目标合约选择保持
    自由。若下一交易日的目标合约与今日不同，就在今日收盘价执行换约，并用该
    规则距离上一次换约的自然日间隔进行年化。

    Args:
        factor_panel: 需含 date, contract, month_tag, f1_ann_basis,
            f3_term_structure, hedge_cost 列（hedge_cost 需调用方提前用
            strategy.hedge_cost(..., N) 算好并挂到面板上）。
        coefs_daily: regression.rolling_ols 的输出，按 date 做过前向填充
            （walk-forward 系数每隔几天才重估一次，展期决策需要每天都有系数可用）。
        N: 对冲成本的展望天数，只用于文档说明，不参与本函数计算
            （hedge_cost 列本身已经是用某个 N 算好的）。

    Returns:
        每条策略一行，含 mean/median/std/p95、最差年份和有效换约次数；按平均
        年化空头展仓成本升序排列。
    """
    month_dummy_cols = [c for c in factor_panel.columns if c.startswith("D_month_")]
    panel = factor_panel.sort_values(["date", "contract"]).copy()
    choices: dict[str, dict[pd.Timestamp, str]] = {name: {} for name in ("当月", "季月", "最低贴水", "回归展仓")}
    prices: dict[pd.Timestamp, dict[str, float]] = {}
    spots: dict[pd.Timestamp, float] = {}

    for d, grp in panel.groupby("date"):
        d = pd.Timestamp(d)
        prices[d] = dict(zip(grp["contract"], grp["F"]))
        spots[d] = float(grp["S"].iloc[0])
        # 目标合约必须至少覆盖下一 N 个交易日；进入到期窗口的合约保留作旧仓
        # 平仓定价，但不能再被任何规则选为新仓。
        eligible = grp[grp["days_to_expiry"] > N].sort_values("days_to_expiry")
        if eligible.empty:
            continue
        choices["当月"][d] = str(eligible.iloc[0]["contract"])
        quarter = eligible[eligible["contract"].str[-2:].astype(int).isin([3, 6, 9, 12])]
        if not quarter.empty:
            choices["季月"][d] = str(quarter.iloc[0]["contract"])
        valid_basis = eligible.dropna(subset=["f1_ann_basis"])
        if not valid_basis.empty:
            choices["最低贴水"][d] = str(valid_basis.loc[valid_basis["f1_ann_basis"].abs().idxmin(), "contract"])
        if not coefs_daily.empty and d in coefs_daily.index:
            coef_row = coefs_daily.loc[d]
            y_hats = []
            for _, row in eligible.iterrows():
                f3 = row["f3_term_structure"] if pd.notna(row["f3_term_structure"]) else 0.0
                dummies = {c: row[c] for c in month_dummy_cols if c in row.index}
                y_hats.append(regression_mod.predict_convergence_rate(coef_row, row["f1_ann_basis"], f3, dummies))
            chosen = best_rollover(eligible.assign(y_hat_convergence_rate=y_hats))
            if chosen is not None:
                choices["回归展仓"][d] = chosen

    all_common_dates = sorted(set.intersection(*(set(v) for v in choices.values())))
    min_dte = panel.groupby("date")["days_to_expiry"].min()
    # 所有规则共享同一调仓网格：每 N 个交易日一次；临近到期窗口加入共同检查日，
    # 避免旧合约在下一常规调仓日前已经消失。合约选择本身仍由各规则决定。
    dates = [
        d for i, d in enumerate(all_common_dates)
        if i % max(N, 1) == 0 or min_dte.get(d, N + 1) <= N
    ]
    rows = []
    event_frames = []
    for name, path in choices.items():
        events = []
        previous_roll_date = dates[0] if dates else None
        held_contract = path[dates[0]] if dates else None
        for d in dates[1:]:
            old_contract, new_contract = held_contract, path[d]
            if old_contract == new_contract:
                continue
            if old_contract not in prices[d] or new_contract not in prices[d] or spots[d] <= 0:
                continue
            interval = (d - previous_roll_date).days if previous_roll_date is not None else np.nan
            points = prices[d][old_contract] - prices[d][new_contract]
            pct = points / spots[d]
            annualized = pct * 365.0 / interval if pd.notna(interval) and interval > 0 else np.nan
            events.append({
                "date": d,
                "old_contract": old_contract,
                "new_contract": new_contract,
                "F_old": prices[d][old_contract],
                "F_new": prices[d][new_contract],
                "S": spots[d],
                "interval_days": interval,
                "points": points,
                "pct": pct,
                "annualized": annualized,
            })
            previous_roll_date = d
            held_contract = new_contract
        event_df = pd.DataFrame(events)
        if not event_df.empty:
            event_frames.append(event_df.assign(策略=name))
        annual = event_df["annualized"].dropna() if not event_df.empty else pd.Series(dtype=float)
        yearly = event_df.dropna(subset=["annualized"]).assign(year=lambda x: pd.to_datetime(x["date"]).dt.year).groupby("year")["annualized"].mean() if not event_df.empty else pd.Series(dtype=float)
        rows.append({
            "策略": name,
            "mean_roll_cost_points": float(event_df["points"].mean()) if not event_df.empty else np.nan,
            "mean_roll_cost_pct": float(event_df["pct"].mean()) if not event_df.empty else np.nan,
            "mean_annualized_cost": float(annual.mean()) if len(annual) else np.nan,
            "median_annualized_cost": float(annual.median()) if len(annual) else np.nan,
            "std_annualized_cost": float(annual.std(ddof=1)) if len(annual) > 1 else 0.0,
            "p95_annualized_cost": float(annual.quantile(0.95)) if len(annual) else np.nan,
            "worst_year": int(yearly.idxmax()) if len(yearly) else pd.NA,
            "worst_year_mean_cost": float(yearly.max()) if len(yearly) else np.nan,
            "n_obs": int(len(event_df)),
            "n_common_evaluation_dates": int(len(dates)),
        })
    ladder = pd.DataFrame(rows).sort_values("mean_annualized_cost", na_position="last")
    ladder = ladder.reset_index(drop=True)
    ladder.attrs["roll_events"] = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(columns=[
            "date", "old_contract", "new_contract", "F_old", "F_new", "S",
            "interval_days", "points", "pct", "annualized", "策略",
        ])
    )
    return ladder


# --------------------------------------------------------------------------- #
# 结构建模：正向 / 反向套利收益率
# --------------------------------------------------------------------------- #

def forward_arb(
    Ft: float,
    St: float,
    Mf: float,
    rf: float,
    Cs: float,
    Cf: float,
    T,
    t,
) -> float:
    """正向套利（买现货、抛期货）收益率。

    P = [(Ft-St) - (St+Ft*Mf)*(1+rf)^((T-t)/360) - St*Cs - Ft*Cf] / (St+Ft*Mf)

    Args:
        Ft: 期货价。St: 现货指数点位。
        Mf: 期货保证金比例。
        rf: 资金成本（无风险利率，年化）。
        Cs: 现货端交易成本比例。Cf: 期货端交易成本比例。
        T: 到期日（date/Timestamp）。t: 当前日（date/Timestamp）。

    Returns:
        以占用资金 (St + Ft*Mf) 计的套利收益率（未年化，公式自带 (T-t)/360 计息）。
    """
    days = _days_between(T, t)
    capital = St + Ft * Mf
    opportunity_cost = capital * (1.0 + rf) ** (days / 360.0)
    numerator = (Ft - St) - opportunity_cost - St * Cs - Ft * Cf
    return numerator / capital


def reverse_arb(
    Ft: float,
    St: float,
    Ml: float,
    Mf: float,
    rf: float,
    rl: float,
    Cs: float,
    Cf: float,
    T,
    t,
) -> float:
    """反向套利（融券卖现货、买期货）收益率。

    分母改为 St*Ml + Ft*Mf（Ml=融券保证金比例），分子在正向套利公式基础上
    额外扣减融券成本 St*rl*(T-t)/360：

    P = [(Ft-St) - (St*Ml+Ft*Mf)*(1+rf)^((T-t)/360) - St*Cs - Ft*Cf - St*rl*(T-t)/360]
        / (St*Ml + Ft*Mf)
    """
    days = _days_between(T, t)
    denom = St * Ml + Ft * Mf
    opportunity_cost = denom * (1.0 + rf) ** (days / 360.0)
    borrow_cost = St * rl * days / 360.0
    numerator = (Ft - St) - opportunity_cost - St * Cs - Ft * Cf - borrow_cost
    return numerator / denom


def _days_between(T, t) -> float:
    T_ts, t_ts = pd.Timestamp(T), pd.Timestamp(t)
    return float((T_ts - t_ts).days)


# --------------------------------------------------------------------------- #
# 【规则回测】：择时信号
# --------------------------------------------------------------------------- #

def causal_basis_thresholds(
    ann_basis: pd.Series,
    contract: pd.Series,
    date: pd.Series,
    window: int,
    pct: float,
    min_periods: int,
) -> pd.DataFrame:
    """仅用 t 之前的观测计算 t 日滚动分位阈值。

    核心是先 ``shift(1)`` 再滚动；因此当天值和任何未来值都不可能进入当天阈值。
    """
    if not 0 < pct < 0.5:
        raise ValueError("pct 必须在 (0, 0.5) 内")
    original = ann_basis.index
    frame = pd.DataFrame({
        "ann_basis": ann_basis.values,
        "contract": contract.values,
        "date": pd.to_datetime(date.values),
    }, index=original).sort_values(["contract", "date"])
    history = frame.groupby("contract")["ann_basis"].shift(1)
    grouped_history = history.groupby(frame["contract"])
    frame["lower_threshold"] = grouped_history.transform(
        lambda s: s.rolling(window, min_periods=min_periods).quantile(pct)
    )
    frame["upper_threshold"] = grouped_history.transform(
        lambda s: s.rolling(window, min_periods=min_periods).quantile(1 - pct)
    )
    return frame[["lower_threshold", "upper_threshold"]].reindex(original)


def fixed_direction_signal(
    ann_basis: pd.Series,
    contract: pd.Series,
    date: pd.Series,
    window: int,
    pct: float,
    min_periods: int,
    rule: Literal["momentum_long", "premium_long", "two_sided"],
) -> pd.Series:
    """三个事前固定方向信号；不读取历史收益，也不在方向之间做选择。"""
    thresholds = causal_basis_thresholds(
        ann_basis, contract, date, window=window, pct=pct, min_periods=min_periods
    )
    discount = ann_basis.le(thresholds["lower_threshold"])
    premium = ann_basis.ge(thresholds["upper_threshold"])
    signal = pd.Series(0, index=ann_basis.index, dtype=int)
    if rule == "momentum_long":
        signal.loc[discount] = 1
    elif rule == "premium_long":
        signal.loc[premium] = 1
    elif rule == "two_sided":
        signal.loc[discount] = 1
        signal.loc[premium] = -1
    else:
        raise ValueError(f"未知固定信号: {rule}")
    return signal

def timing_signal(
    ann_basis: pd.Series,
    contract: pd.Series,
    date: pd.Series,
    window: int,
    pct: float,
    mode: Literal["momentum", "reversal"] = "reversal",
) -> pd.Series:
    """【规则回测】基于年化基差滚动分位数的择时信号，不是被检验过的alpha模型。

    对每个 contract 分组、按 date 排序后，用过去 window 个交易日的年化基差
    计算 pct 分位与 (1-pct) 分位的滚动阈值：
        - mode='reversal' （均值回复假设）：
            ann_basis 跌破下分位（深度贴水）-> +1（买入，赌收敛）
            ann_basis 突破上分位（深度升水）-> -1（卖出，赌收敛）
        - mode='momentum' （动量假设）：符号与 reversal 相反。

    Returns:
        与 ann_basis 对齐的信号 Series，取值 {-1, 0, 1}。
    """
    orig_index = ann_basis.index
    df = pd.DataFrame(
        {"ann_basis": ann_basis.values, "contract": contract.values, "date": date.values},
        index=orig_index,
    )
    df = df.sort_values(["contract", "date"])

    grouped = df.groupby("contract")["ann_basis"]
    df["lo"] = grouped.transform(lambda s: s.rolling(window, min_periods=window).quantile(pct))
    df["hi"] = grouped.transform(lambda s: s.rolling(window, min_periods=window).quantile(1 - pct))

    df["signal"] = 0
    df.loc[df["ann_basis"] < df["lo"], "signal"] = 1
    df.loc[df["ann_basis"] > df["hi"], "signal"] = -1

    if mode == "momentum":
        df["signal"] = -df["signal"]
    elif mode != "reversal":
        raise ValueError(f"未知 mode: {mode!r}，只支持 'momentum' / 'reversal'")

    return df["signal"].reindex(orig_index)


def map_by_index(
    index_code: str,
    ann_basis: pd.Series,
    contract: pd.Series,
    date: pd.Series,
    window: int,
    pct: float,
) -> pd.Series:
    """【规则回测】按标的指数特性映射基差择时信号，均为均值回复假设下的规则：

        - IH(000016) / IF(000300)：长期温和贴水，"大幅贴水"（跌破下分位）视为
          异常偏离 -> 买入（做多基差，赌贴水收敛）。
        - IC(000905) / IM(000852)：结构性深度贴水品种，"大幅升水"（突破上分位，
          即贴水显著收窄甚至转正）视为异常偏离 -> 买入。

    这两条规则本质上都是"押注基差向该品种历史中枢回归"，只是触发方向不同。
    """
    orig_index = ann_basis.index
    df = pd.DataFrame(
        {"ann_basis": ann_basis.values, "contract": contract.values, "date": date.values},
        index=orig_index,
    )
    df = df.sort_values(["contract", "date"])

    grouped = df.groupby("contract")["ann_basis"]
    df["lo"] = grouped.transform(lambda s: s.rolling(window, min_periods=window).quantile(pct))
    df["hi"] = grouped.transform(lambda s: s.rolling(window, min_periods=window).quantile(1 - pct))

    signal = pd.Series(0, index=df.index, dtype=int)
    if index_code in ("000016", "000300"):
        signal[df["ann_basis"] < df["lo"]] = 1
    elif index_code in ("000905", "000852"):
        signal[df["ann_basis"] > df["hi"]] = 1
    else:
        raise ValueError(f"未知 index_code: {index_code!r}")

    return signal.reindex(orig_index)


def backtest_position_series(
    dates: pd.Series,
    signal: pd.Series,
    ret_1d: pd.Series,
    N: int,
    cost: float = 0.0,
    initial_capital: float = 1.0,
    position_override: Optional[pd.Series] = None,
) -> dict:
    """【规则回测，主推荐】仓位序列法：每日只结算 1 日收益，不重叠、不重复计数。

    这是修复 backtest_naive_overlapping() 年化虚高问题的正确做法：不再把重叠的
    N 日前瞻收益天天复利，而是先把信号转换成一条"目标仓位"时间序列，再用真正
    互不重叠的单日收益逐日结算：

        position(t) = signal 中最近一次非零取值，向前填充最多 N-1 天
                      （即"持有 N 个交易日"），裁剪到 [-1, 1]。
        daily_pnl(t) = position(t) * ret_1d(t) - cost * |position(t) - position(t-1)|

    只要 ret_1d 是"从 t 到 t+1"的单日收益（不是重叠的 N 日收益），同一段行情
    就只会在它真实发生的那一天被计入一次，逐日复利不会放大出不可交易的收益率。

    Args:
        dates: 日期序列（用于排序与年化计算，需与 signal/ret_1d 等长对齐）。
        signal: 择时信号，一般是 {-1,0,1}，也允许连续值（会被裁剪到 [-1,1]）。
        ret_1d: 单日前瞻收益，"从 t 到 t+1"的口径，例如用 N=1 调用
            factors.basis_convergence_rate(adj_basis, S, contract, date, N=1) 得到。
            **不要**传入 N>1 的重叠窗口收益，否则又会重新引入双重计数问题。
        N: 每次信号的目标持有天数，用于构造 position 的前向填充窗口。
        cost: 每次仓位变动的交易成本比例，按 |Δposition| 线性收取，默认 0。
        initial_capital: 初始净值，默认 1.0。
        position_override: 可选的已构造仓位路径。用于 walk-forward 分窗口构造仓位、
            再统一计算 N 阶 HAC 指标；传入时不再对 signal 做持有期前向填充。

    Returns:
        dict:
            equity_curve: DataFrame[date, equity]
            annualized_return: float（按 252 交易日年化）
            annualized_vol: float（日收益标准差 × sqrt(252)）
            sharpe: float（年化收益 / 年化波动，波动为 0 时记为 0）
            sharpe_nw: float（用 Newey-West 长期方差估计替代普通标准差算出的
                自相关调整版 Sharpe，持仓期带来的日收益序列相关性会拉低它，
                比 sharpe 更保守，二者差距大说明普通 Sharpe 可能虚高）
            max_drawdown: float
            win_rate: float（有持仓的交易日中，当日 daily_pnl > 0 的占比；
                完全没有持仓时记为 NaN）
            n_trades: int（仓位发生变化的次数）
            signal_tstat_hac: float（策略日收益均值是否显著不为 0 的 Newey-West
                HAC t 值，滞后阶 = N，用于在持仓期induced的自相关下不高估显著性；
                样本量不足以支撑给定滞后阶时返回 NaN）
    """
    df = pd.DataFrame({"date": dates.values, "signal": signal.values, "ret_1d": ret_1d.values})
    if position_override is not None:
        if len(position_override) != len(df):
            raise ValueError("position_override 必须与 dates 等长")
        df["position_override"] = position_override.values
    if df["date"].duplicated().any():
        raise ValueError(
            "backtest_position_series 要求每个交易日只有一条观测；"
            "请先选定连续合约/展仓规则，不能把同日多合约当成多个交易日复利。"
        )
    df = df.sort_values("date").reset_index(drop=True)

    hold_window = max(N - 1, 0)
    if position_override is None:
        position = df["signal"].where(df["signal"] != 0, other=np.nan)
        position = position.ffill(limit=hold_window).fillna(0.0).clip(-1.0, 1.0)
    else:
        position = df["position_override"].fillna(0.0).clip(-1.0, 1.0)
    df["position"] = position

    position_change = df["position"].diff()
    position_change.iloc[0] = df["position"].iloc[0] if len(df) else 0.0
    turnover = position_change.abs()

    df["daily_pnl"] = df["position"] * df["ret_1d"] - cost * turnover
    df = df.dropna(subset=["daily_pnl"]).reset_index(drop=True)

    if df.empty:
        return {
            "equity_curve": pd.DataFrame(columns=["date", "equity"]),
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "sharpe_nw": 0.0,
            "max_drawdown": 0.0,
            "win_rate": float("nan"),
            "n_trades": 0,
            "signal_tstat_hac": float("nan"),
        }

    df["equity"] = initial_capital * (1.0 + df["daily_pnl"]).cumprod()

    n_periods = len(df)
    total_return = df["equity"].iloc[-1] / initial_capital - 1.0
    n_years = max(n_periods / 252.0, 1e-9)
    annualized_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0
    annualized_vol = float(df["daily_pnl"].std(ddof=1) * np.sqrt(252)) if n_periods > 1 else 0.0
    sharpe = float(annualized_return / annualized_vol) if annualized_vol > 0 else 0.0

    running_max = df["equity"].cummax()
    drawdown = df["equity"] / running_max - 1.0
    max_drawdown = float(drawdown.min())

    held = df[df["position"] != 0]
    win_rate = float((held["daily_pnl"] > 0).mean()) if len(held) > 0 else float("nan")

    n_trades = int((df["position"].diff().fillna(df["position"]) != 0).sum())

    signal_tstat_hac = float("nan")
    sharpe_nw = 0.0
    if n_periods > N + 1:
        X = np.ones((n_periods, 1))
        hac_model = sm.OLS(df["daily_pnl"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": N})
        signal_tstat_hac = float(hac_model.tvalues[0])

        hac_se_of_mean = float(hac_model.bse[0])
        long_run_std = hac_se_of_mean * np.sqrt(n_periods)
        mean_daily_pnl = float(df["daily_pnl"].mean())
        sharpe_nw = (mean_daily_pnl / long_run_std * np.sqrt(252)) if long_run_std > 0 else 0.0

    return {
        "equity_curve": df[["date", "equity"]],
        "annualized_return": float(annualized_return),
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "sharpe_nw": float(sharpe_nw),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "signal_tstat_hac": signal_tstat_hac,
    }


def backtest_naive_overlapping(
    dates: pd.Series,
    signal: pd.Series,
    forward_return: pd.Series,
    initial_capital: float = 1.0,
) -> dict:
    """【规则回测，illustrative only —— 收益被重叠窗口放大，不可用于横向对比】

    用 signal(t) * forward_return(t) 逐日复利核算净值曲线，是本模块最早的朴素
    实现。如果 forward_return 来自重叠的 N 日窗口（比如 factors.y_convergence_rate，
    每天都滚动取未来 N 日），同一段行情会被重复计入多次，年化收益率会被显著
    放大且不可交易（历史上曾观测到 +3892% 这类失真数值）。

    保留这个函数只是为了兼容旧代码/做示意用途；**任何需要横向比较不同规则
    优劣的场景，一律改用 backtest_position_series()**，它才是可交易表述、
    不重复计数的正确回测方法。

    Args:
        dates: 与 signal/forward_return 对齐的日期序列。
        signal: {-1,0,1} 仓位信号（当期持仓方向）。
        forward_return: 该期信号对应的未来收益率（如重叠的 N 日
            y_convergence_rate），由调用方按场景定义。
        initial_capital: 初始净值，默认 1.0。

    Returns:
        dict:
            equity_curve: DataFrame[date, equity]
            annualized_return: float
            max_drawdown: float
            n_trades: int（信号发生变化的次数，粗略代理换仓次数）
            note: str，固定提示 "illustrative only — 收益被重叠窗口放大，不可用于横向对比"
    """
    NOTE = "illustrative only — 收益被重叠窗口放大，不可用于横向对比"

    df = pd.DataFrame({"date": dates.values, "signal": signal.values, "fwd_ret": forward_return.values})
    df = df.sort_values("date").dropna(subset=["signal", "fwd_ret"]).reset_index(drop=True)
    if df.empty:
        return {
            "equity_curve": pd.DataFrame(columns=["date", "equity"]),
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "note": NOTE,
        }

    df["period_return"] = df["signal"] * df["fwd_ret"]
    df["equity"] = initial_capital * (1.0 + df["period_return"]).cumprod()

    n_periods = len(df)
    total_return = df["equity"].iloc[-1] / initial_capital - 1.0
    n_years = max(n_periods / 252.0, 1e-9)
    annualized_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0

    running_max = df["equity"].cummax()
    drawdown = df["equity"] / running_max - 1.0
    max_drawdown = float(drawdown.min())

    n_trades = int((df["signal"].diff().fillna(0) != 0).sum())

    return {
        "equity_curve": df[["date", "equity"]],
        "annualized_return": float(annualized_return),
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "note": NOTE,
    }
