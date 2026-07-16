"""data.py —— 数据层。

方法论性质：**规则接入 / 工程适配**（本身不构成金融模型，只负责把外部数据整理成
统一 schema）。下游 dividend / factors / regression / strategy / forecast 一律只
认这一层输出的面板格式，不直接碰原始数据源。

面板 schema（长表，一行 = 一个合约在一个交易日的观测）：
    date            : pd.Timestamp   交易日
    contract        : str            合约代码，如 "IF2409"
    index_code      : str            标的指数代码（000016/000300/000905/000852）
    month_tag       : str            当月 / 次月 / 季月 / 下季月
    F               : float          期货价（收盘价，近似 mark price）
    S               : float          现货指数收盘价
    days_to_expiry  : int            距到期日的剩余交易日 D
    calendar_days_to_expiry : int    距到期日的剩余自然日（备用审计口径）

设计要点：
    - DataAdapter 是抽象基类，规定 load_futures / load_index / load_dividends 三个
      接口，任何具体数据源（AkShare、聚宽、本地库……）都通过实现这三个方法接入。
    - AkShareAdapter 是"真实数据"实现，但 akshare 的接口名/字段极易随版本变化，
      因此每次调用都包裹 try/except + 重试，失败就打印接口名并向下游返回 None，
      由调用方决定是否回退到本地缓存或 MockAdapter。
    - MockAdapter 只生成中性的可复现模拟面板，用于单元测试和流水线演示；
      不植入分红季、指数风格或策略方向等研究结论。
    - ParquetCache 是极简本地缓存（parquet，pyarrow 不可用时退化为 csv），
      key 由 index_code + start + end + kind 组成，refresh=True 强制重取。
"""

from __future__ import annotations

import abc
import hashlib
import io
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# 常量 / 映射表
# --------------------------------------------------------------------------- #

#: 指数代码 -> 期货品种代码（中金所）
INDEX_TO_PRODUCT: dict[str, str] = {
    "000016": "IH",  # 上证50
    "000300": "IF",  # 沪深300
    "000905": "IC",  # 中证500
    "000852": "IM",  # 中证1000
}

PRODUCT_TO_INDEX: dict[str, str] = {v: k for k, v in INDEX_TO_PRODUCT.items()}

#: 面板 schema 列顺序，供各模块统一引用
PANEL_COLUMNS: list[str] = [
    "date",
    "contract",
    "index_code",
    "month_tag",
    "F",
    "S",
    "days_to_expiry",
    "calendar_days_to_expiry",
]

_CFFEX_TRADE_DATES: pd.DatetimeIndex | None = None


def configure_cffex_trade_calendar(trade_dates) -> None:
    """配置中金所交易日集合；本地数据可直接用自身完整日期列配置。"""
    global _CFFEX_TRADE_DATES
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(trade_dates), errors="coerce").dropna())
    _CFFEX_TRADE_DATES = dates.normalize().unique().sort_values()

MONTH_TAG_ORDER: list[str] = ["当月", "次月", "季月", "下季月"]

QUARTER_MONTHS = {3, 6, 9, 12}

_CONTRACT_RE = re.compile(r"^(?P<product>[A-Za-z]{1,2})(?P<yymm>\d{4})")


# --------------------------------------------------------------------------- #
# 到期日工具：中金所股指期货到期日 = 合约月份第三个周五
# --------------------------------------------------------------------------- #

def parse_contract(contract: str) -> tuple[str, int, int]:
    """解析合约代码，返回 (品种代码, 年, 月)。

    支持 "IF2409" / "if2409" / "IF2409.CFX" 等写法。
    """
    m = _CONTRACT_RE.match(contract.strip())
    if not m:
        raise ValueError(f"无法解析合约代码: {contract!r}")
    product = m.group("product").upper()
    yymm = m.group("yymm")
    yy, mm = int(yymm[:2]), int(yymm[2:])
    if not 1 <= mm <= 12:
        raise ValueError(f"合约代码月份非法: {contract!r}")
    year = 2000 + yy
    return product, year, mm


def third_friday(year: int, month: int) -> date:
    """返回给定年月的第三个周五。"""
    first_of_month = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4 ... Sunday=6
    days_to_first_friday = (4 - first_of_month.weekday()) % 7
    first_friday = first_of_month + timedelta(days=days_to_first_friday)
    return first_friday + timedelta(weeks=2)


def cffex_expiry(contract: str, trade_dates=None) -> date:
    """返回中金所最后交易日：第三个周五，休市则顺延到下一交易日。

    ``trade_dates`` 可显式传入；否则使用 ``configure_cffex_trade_calendar``
    配置的日历。没有可用日历时仅退化为工作日判断，调用方不应把该退化口径
    用于正式历史回测。
    """
    _, year, month = parse_contract(contract)
    scheduled = third_friday(year, month)
    calendar = trade_dates if trade_dates is not None else _CFFEX_TRADE_DATES
    if calendar is not None and len(calendar):
        dates = calendar if calendar is _CFFEX_TRADE_DATES else pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().sort_values()
        scheduled_ts = pd.Timestamp(scheduled)
        pos = int(dates.searchsorted(scheduled_ts, side="left"))
        if pos < len(dates) and dates[pos] <= scheduled_ts + pd.Timedelta(days=14):
            return dates[pos].date()
    candidate = scheduled
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def calendar_days_to_expiry(trade_date: pd.Timestamp | date, contract: str) -> int:
    """剩余自然日。"""
    expiry = cffex_expiry(contract)
    if isinstance(trade_date, pd.Timestamp):
        trade_date = trade_date.date()
    return (expiry - trade_date).days


def days_to_expiry(trade_date: pd.Timestamp | date, contract: str) -> int:
    """剩余交易日数，不含估值日、包含最后交易日。"""
    trade_ts = pd.Timestamp(trade_date).normalize()
    expiry_ts = pd.Timestamp(cffex_expiry(contract))
    if expiry_ts < trade_ts:
        return -1
    if (
        _CFFEX_TRADE_DATES is not None
        and len(_CFFEX_TRADE_DATES)
        and trade_ts >= _CFFEX_TRADE_DATES.min()
        and expiry_ts <= _CFFEX_TRADE_DATES.max()
    ):
        left = int(_CFFEX_TRADE_DATES.searchsorted(trade_ts, side="right"))
        right = int(_CFFEX_TRADE_DATES.searchsorted(expiry_ts, side="right"))
        return right - left
    return int(np.busday_count(trade_ts.date(), expiry_ts.date()))


def assign_month_tags(contracts: list[str]) -> dict[str, str]:
    """给同一交易日内在市的合约打上 当月/次月/季月/下季月 标签。

    规则：按到期日升序排列，前两个记为 当月/次月；剩余合约中优先选取
    "季月"（3/6/9/12 月）排在后面的记为 季月/下季月；若数量不足 4 个，
    多出的标签留空。这是对中金所"当月+次月+两个季月"标准挂牌规则的近似实现。
    """
    if not contracts:
        return {}
    parsed = []
    for c in contracts:
        _, year, month = parse_contract(c)
        parsed.append((cffex_expiry(c), c, year, month))
    parsed.sort(key=lambda x: x[0])

    tags: dict[str, str] = {}
    if len(parsed) >= 1:
        tags[parsed[0][1]] = "当月"
    if len(parsed) >= 2:
        tags[parsed[1][1]] = "次月"

    remaining = parsed[2:]
    quarter_remaining = [p for p in remaining if p[3] in QUARTER_MONTHS]
    non_quarter_remaining = [p for p in remaining if p[3] not in QUARTER_MONTHS]
    ordered_remaining = quarter_remaining + non_quarter_remaining

    quarter_labels = ["季月", "下季月"]
    for label, item in zip(quarter_labels, ordered_remaining):
        tags[item[1]] = label

    return tags


def _calendar_listed_contracts(trade_date: pd.Timestamp | date, product: str) -> list[str]:
    """给定交易日，按 CFFEX"当月+次月+两个季月"挂牌规则，纯日历推导出当天最多
    4 个在市合约代码——不读取任何实际行情，只用到期日规则计算。

    关键点：候选月份必须先过滤掉"本月第三个周五已经过去"的月份，再取最近的
    两个作为当月/次月 —— 否则在每个月第三个周五之后（约占每月后 1/3~2/5 的
    交易日），当月这个 tag 会持续指向一个实际上已经到期的合约代码，被下游
    `days_to_expiry<=0` 过滤掉整行，导致"当月"标签系统性地比其余三个标签
    少了近一半的观测（这是实际发生过的一个 bug，由 hedge_cost_ladder 的
    当月 vs 季月覆盖率对比意外暴露出来）。
    """
    y, m = trade_date.year, trade_date.month
    trade_date_ = trade_date.date() if hasattr(trade_date, "date") else trade_date

    def code(year: int, month: int) -> str:
        return f"{product}{year % 100:02d}{month:02d}"

    months_ahead = []
    yy, mm = y, m
    for _ in range(16):
        months_ahead.append((yy, mm))
        mm += 1
        if mm > 12:
            mm = 1
            yy += 1

    # 只保留"第三个周五还没到"的月份，这样第一个元素才是真正还在交易的当月合约。
    unexpired = [
        (yy_, mm_) for (yy_, mm_) in months_ahead if third_friday(yy_, mm_) > trade_date_
    ]

    current_month, next_month = unexpired[0], unexpired[1]
    quarter_candidates = [
        (yy_, mm_) for (yy_, mm_) in unexpired if mm_ in QUARTER_MONTHS
    ]
    quarter_candidates = [
        q for q in quarter_candidates if q not in (current_month, next_month)
    ]
    quarters = quarter_candidates[:2]

    chosen = [current_month, next_month] + quarters
    # 去重且保持顺序
    seen = set()
    uniq = []
    for q in chosen:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return [code(yy_, mm_) for (yy_, mm_) in uniq]


def month_tag_contract_codes(trade_date: pd.Timestamp | date, product: str) -> dict[str, str]:
    """给定交易日与品种，返回 {月份标签: 合约代码}（标签与 assign_month_tags 一致：
    当月/次月/季月/下季月）。纯日历推导，不依赖任何实际行情。

    两个使用场景：
        - MockAdapter 造模拟数据的合约列表；
        - LocalContinuousAdapter 给"已经按月份标签滚动拼接好的连续合约"行情
          （比如 Wind 导出的"当月连续/次月连续/当季连续/下季连续"）反推
          days_to_expiry —— 连续合约本身不带合约代码，但在 CFFEX 挂牌规则下，
          任意交易日"当月"对应哪个合约代码是确定性的日历函数，可以反推出来，
          年化基差公式（factors.ann_basis）因此不需要为连续合约改写。
    """
    codes = _calendar_listed_contracts(trade_date, product)
    return {tag: c for c, tag in assign_month_tags(codes).items()}


# --------------------------------------------------------------------------- #
# 本地缓存层
# --------------------------------------------------------------------------- #

class ParquetCache:
    """极简本地缓存：优先 parquet（需要 pyarrow），失败退化为 csv。"""

    def __init__(self, cache_dir: str | Path = ".basis_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str, ext: str) -> Path:
        safe_key = re.sub(r"[^A-Za-z0-9_\-.]", "_", key)
        return self.cache_dir / f"{safe_key}.{ext}"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        pq_path = self._key_to_path(key, "parquet")
        csv_path = self._key_to_path(key, "csv")
        try:
            if pq_path.exists():
                return pd.read_parquet(pq_path)
        except Exception as exc:  # pyarrow 未安装或文件损坏
            print(f"[cache] 读取 parquet 失败 ({key}): {exc}")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                for col in df.columns:
                    if col == "date" or col.endswith("_date"):
                        df[col] = pd.to_datetime(df[col])
                return df
            except Exception as exc:
                print(f"[cache] 读取 csv 失败 ({key}): {exc}")
        return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        pq_path = self._key_to_path(key, "parquet")
        try:
            df.to_parquet(pq_path, index=False)
            return
        except Exception as exc:
            print(f"[cache] 写入 parquet 失败，退化为 csv ({key}): {exc}")
        csv_path = self._key_to_path(key, "csv")
        df.to_csv(csv_path, index=False)

    def make_key(self, kind: str, index_code: str, start_date: str, end_date: str) -> str:
        return f"{kind}_{index_code}_{start_date}_{end_date}"


# --------------------------------------------------------------------------- #
# 抽象基类
# --------------------------------------------------------------------------- #

class DataAdapter(abc.ABC):
    """数据适配器抽象基类。所有具体数据源都必须实现这三个接口。"""

    @abc.abstractmethod
    def load_futures(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        """加载某指数对应期货品种的全合约日行情。

        返回列至少包含: date, contract, index_code, month_tag, F, days_to_expiry
        （S 由 load_index 结果合并补齐，不在本方法内提供）。
        """

    @abc.abstractmethod
    def load_index(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        """加载现货指数日收盘价，返回列: date, index_code, S。"""

    @abc.abstractmethod
    def load_dividends(
        self, index_code: str, year: int, refresh: bool = False
    ) -> pd.DataFrame:
        """加载成分股分红预测面板。

        返回列: stock_code, index_code, ex_date, dividend_per_share,
        weight_pct, dividend_yield。
        """


# --------------------------------------------------------------------------- #
# MockAdapter：离线功能测试数据（不表达市场观点）
# --------------------------------------------------------------------------- #

class MockAdapter(DataAdapter):
    """中性的离线模拟数据源，只用于测试接口和计算流水线。

    不包含固定分红季效应，不预设 IC/IM 比 IH/IF 贴水更深，也不应被用于
    判断因子有效性、策略收益或市场统计规律。
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

    def _rng_for(self, *parts: object) -> np.random.Generator:
        """按请求参数构造稳定随机数流，保证期货和现货重复加载时路径一致。"""
        payload = "|".join(map(str, (self.seed, *parts))).encode("utf-8")
        stable_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return np.random.default_rng(stable_seed)

    # -- 期货 ---------------------------------------------------------------
    def load_futures(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        panel = self.generate_mock_panel(index_code, start_date, end_date)
        return panel[[
            "date", "contract", "index_code", "month_tag", "F",
            "days_to_expiry", "calendar_days_to_expiry",
        ]]

    # -- 现货指数 -------------------------------------------------------------
    def load_index(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        panel = self.generate_mock_panel(index_code, start_date, end_date)
        idx = panel[["date", "index_code", "S"]].drop_duplicates(subset=["date", "index_code"])
        return idx.reset_index(drop=True)

    # -- 分红 -----------------------------------------------------------------
    def load_dividends(
        self, index_code: str, year: int, refresh: bool = False
    ) -> pd.DataFrame:
        rng = self._rng_for("dividends", index_code, year)
        n_stocks = 20
        rows = []
        for i in range(n_stocks):
            # 除息高峰集中在 5-7 月
            month = int(rng.choice([5, 6, 6, 6, 7, 7, 8], size=1)[0])
            day = int(rng.integers(1, 28))
            ex_date = pd.Timestamp(year=year, month=month, day=day)
            dividend_per_share = float(rng.uniform(0.1, 1.2))
            weight_pct = float(rng.uniform(0.2, 5.0))
            price = float(rng.uniform(8, 80))
            rows.append(
                {
                    "stock_code": f"MOCK{i:04d}",
                    "index_code": index_code,
                    "ex_date": ex_date,
                    "dividend_per_share": dividend_per_share,
                    "weight_pct": weight_pct,
                    "dividend_yield": dividend_per_share / price,
                }
            )
        return pd.DataFrame(rows)

    # -- 核心生成逻辑 -----------------------------------------------------------
    def generate_mock_panel(
        self, index_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """生成零均值基差扰动的模拟面板（IH/IF/IC/IM 通用）。"""
        product = INDEX_TO_PRODUCT.get(index_code)
        if product is None:
            raise ValueError(f"未知 index_code: {index_code}")

        dates = pd.bdate_range(start=start_date, end=end_date)
        if len(dates) == 0:
            return pd.DataFrame(columns=PANEL_COLUMNS)

        rng = self._rng_for("panel", index_code, start_date, end_date)
        s0 = {"IH": 2600.0, "IF": 3700.0, "IC": 5800.0, "IM": 6200.0}[product]
        # 现货指数：几何随机游走
        rets = rng.normal(loc=0.0002, scale=0.011, size=len(dates))
        s_path = s0 * np.cumprod(1 + rets)

        # 缓慢均值回复的零均值共同状态，只为测试提供连续性，不编码方向或季节规律。
        ann_basis_state = np.zeros(len(dates))
        for i in range(1, len(dates)):
            ann_basis_state[i] = 0.97 * ann_basis_state[i - 1] + rng.normal(0.0, 0.0015)

        rows = []
        for i, d in enumerate(dates):
            contracts = self._listed_contracts(d, product)
            tags = assign_month_tags(contracts)
            S = float(s_path[i])
            for c in contracts:
                D = days_to_expiry(d, c)
                calendar_D = calendar_days_to_expiry(d, c)
                if D <= 0:
                    continue
                tag = tags.get(c, "")
                ann_basis = ann_basis_state[i] + rng.normal(0.0, 0.001)
                F = S * (1 + ann_basis * D / 360.0)
                rows.append(
                    {
                        "date": d,
                        "contract": c,
                        "index_code": index_code,
                        "month_tag": tag,
                        "F": F,
                        "S": S,
                        "days_to_expiry": D,
                        "calendar_days_to_expiry": calendar_D,
                    }
                )
        return pd.DataFrame(rows, columns=PANEL_COLUMNS)

    @staticmethod
    def _listed_contracts(trade_date: pd.Timestamp, product: str) -> list[str]:
        """给定交易日，构造"当月+次月+两个季月"共4个在市合约代码。

        日历推导逻辑已提升为模块级函数 `_calendar_listed_contracts`（供
        LocalContinuousAdapter 复用），本方法只是薄包装。历史 bug 注记见该函数
        docstring：候选月份必须先过滤掉"本月第三个周五已经过去"的月份，否则
        "当月"标签会系统性地比其余三个标签少掉近一半观测。
        """
        return _calendar_listed_contracts(trade_date, product)


# --------------------------------------------------------------------------- #
# 共用工具：期货明细 -> 面板（打 days_to_expiry / month_tag）
# --------------------------------------------------------------------------- #

def _tag_futures_panel(raw: pd.DataFrame, index_code: str) -> pd.DataFrame:
    """把整理好的期货明细（至少含列 date/contract/F）逐日打 days_to_expiry 与
    month_tag，产出 PANEL_COLUMNS 顺序的面板（S 留空，由 merge_panel 补齐）。

    AkShareAdapter / CffexAkShareAdapter / LocalFileAdapter 的 load_futures 共用
    这段"先剔除当日已到期合约、再排当月/次月"的逻辑，避免同一段易错代码抄三份
    （历史上 assign_month_tags 的调用顺序错误导致过"当月"标签系统性缺失的 bug，
    见 MockAdapter._listed_contracts 的注记）。
    """
    rows = []
    for d, grp in raw.groupby("date"):
        valid_contracts = []
        dte_by_contract: dict[str, int] = {}
        calendar_dte_by_contract: dict[str, int] = {}
        for contract in sorted(grp["contract"].unique().tolist()):
            try:
                dte_by_contract[contract] = days_to_expiry(d, contract)
                calendar_dte_by_contract[contract] = calendar_days_to_expiry(d, contract)
            except ValueError:
                continue
            if dte_by_contract[contract] > 0:
                valid_contracts.append(contract)
        tags = assign_month_tags(valid_contracts)
        for row in grp.itertuples(index=False):
            D = dte_by_contract.get(row.contract, -1)
            if D <= 0:
                continue
            rows.append(
                {
                    "date": d,
                    "contract": row.contract,
                    "index_code": index_code,
                    "month_tag": tags.get(row.contract, ""),
                    "F": row.F,
                    "S": np.nan,
                    "days_to_expiry": D,
                    "calendar_days_to_expiry": calendar_dte_by_contract.get(row.contract, -1),
                }
            )
    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


# --------------------------------------------------------------------------- #
# AkShareAdapter：真实数据源
# --------------------------------------------------------------------------- #

def _retry(func, retries: int = 2, delay: float = 1.0, name: str = ""):
    """对外部接口调用做简单重试封装，失败打印接口名并返回 None。"""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:  # akshare 接口/字段随版本变化，宽泛捕获
            last_exc = exc
            if attempt < retries:
                time.sleep(delay)
    print(f"[akshare] 接口调用失败: {name!r} -> {last_exc!r}")
    return None


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """在 df.columns 中按候选名单（含子串匹配）找第一个命中的列名。

    akshare 的列名经常随版本变化（如"收盘价"/"close"/"收盘"），用这个
    兜底查找来降低耦合。
    """
    cols = list(df.columns)
    lower_to_original = {str(col).strip().lower(): col for col in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    for cand in candidates:
        for col in cols:
            if cand.lower() in str(col).lower():
                return col
    return None


@dataclass
class AkShareAdapter(DataAdapter):
    """真实数据源：基于 akshare。

    akshare 的接口名 / 返回字段会随版本变化，因此本类：
        1. 每个外部调用都包一层 try/except + 重试 (_retry)；
        2. 失败时打印具体接口名，方便定位是哪个 akshare 接口挂了；
        3. 调用方（如 run.py）应在拿到空结果 / 异常时自行回退到本地缓存或
           MockAdapter，本类自身只负责"尽力而为"地拿真实数据。

    可通过 futures_interface_name / index_interface_name 等字段配置替换成
    其他版本的 akshare 接口名，无需改动业务逻辑代码。
    """

    cache_dir: str | Path = ".basis_cache"
    retries: int = 2
    retry_delay: float = 1.0
    # 接口名可配置，便于 akshare 升级后替换
    futures_interface_name: str = "get_futures_daily"
    index_interface_name: str = "index_zh_a_hist"
    index_fallback_interface_name: str = "stock_zh_index_hist_csindex"
    dividend_interface_name: str = "stock_fhps_em"
    weight_interface_name: str = "index_stock_cons_weight_csindex"

    def __post_init__(self) -> None:
        self.cache = ParquetCache(self.cache_dir)

    # -- 期货 ---------------------------------------------------------------
    def load_futures(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        key = self.cache.make_key("futures", index_code, start_date, end_date)
        if not refresh:
            cached = self.cache.get(key)
            if cached is not None and not cached.empty:
                return cached

        product = INDEX_TO_PRODUCT.get(index_code)
        if product is None:
            raise ValueError(f"未知 index_code: {index_code}")

        def _call() -> pd.DataFrame:
            import akshare as ak

            fn = getattr(ak, self.futures_interface_name)
            df = fn(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                market="CFFEX",
            )
            return df

        raw = _retry(
            _call, retries=self.retries, delay=self.retry_delay, name=self.futures_interface_name
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=PANEL_COLUMNS)

        sym_col = _first_matching_column(raw, ["symbol", "合约代码", "合约"])
        date_col = _first_matching_column(raw, ["date", "交易日期", "日期"])
        close_col = _first_matching_column(raw, ["close", "收盘价", "收盘"])
        if sym_col is None or date_col is None or close_col is None:
            print(
                f"[akshare] {self.futures_interface_name} 返回字段无法识别: {list(raw.columns)}"
            )
            return pd.DataFrame(columns=PANEL_COLUMNS)

        raw = raw[raw[sym_col].astype(str).str.upper().str.startswith(product)].copy()
        if raw.empty:
            return pd.DataFrame(columns=PANEL_COLUMNS)

        raw["date"] = pd.to_datetime(raw[date_col].astype(str))
        raw["contract"] = raw[sym_col].astype(str).str.upper()
        raw["F"] = pd.to_numeric(raw[close_col], errors="coerce")

        result = _tag_futures_panel(raw[["date", "contract", "F"]], index_code)
        if not result.empty:
            self.cache.put(key, result)
        return result

    # -- 现货指数 -------------------------------------------------------------
    def load_index(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        key = self.cache.make_key("index", index_code, start_date, end_date)
        if not refresh:
            cached = self.cache.get(key)
            if cached is not None and not cached.empty:
                return cached

        def _call(interface_name: str) -> pd.DataFrame:
            import akshare as ak

            fn = getattr(ak, interface_name)
            kwargs = {
                "symbol": index_code,
                "start_date": start_date.replace("-", ""),
                "end_date": end_date.replace("-", ""),
            }
            if interface_name == self.index_interface_name:
                kwargs["period"] = "daily"
            return fn(**kwargs)

        raw = _retry(
            lambda: _call(self.index_interface_name),
            retries=self.retries,
            delay=self.retry_delay,
            name=self.index_interface_name,
        )
        interface_used = self.index_interface_name
        if raw is None or raw.empty:
            raw = _retry(
                lambda: _call(self.index_fallback_interface_name),
                retries=self.retries,
                delay=self.retry_delay,
                name=self.index_fallback_interface_name,
            )
            interface_used = self.index_fallback_interface_name
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date", "index_code", "S"])

        date_col = _first_matching_column(raw, ["date", "日期"])
        close_col = _first_matching_column(raw, ["close", "收盘", "收盘价"])
        if date_col is None or close_col is None:
            print(
                f"[akshare] {interface_used} 返回字段无法识别: {list(raw.columns)}"
            )
            return pd.DataFrame(columns=["date", "index_code", "S"])

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_col].astype(str)),
                "index_code": index_code,
                "S": pd.to_numeric(raw[close_col], errors="coerce"),
            }
        )
        out = out.dropna(subset=["S"]).reset_index(drop=True)
        if not out.empty:
            self.cache.put(key, out)
        return out

    # -- 分红 -----------------------------------------------------------------
    def load_dividends(
        self, index_code: str, year: int, refresh: bool = False
    ) -> pd.DataFrame:
        key = self.cache.make_key("dividend", index_code, str(year), str(year))
        if not refresh:
            cached = self.cache.get(key)
            if cached is not None and not cached.empty:
                return cached

        def _call_fhps() -> pd.DataFrame:
            import akshare as ak

            fn = getattr(ak, self.dividend_interface_name)
            return fn(date=f"{year}1231")

        def _call_weight() -> pd.DataFrame:
            import akshare as ak

            fn = getattr(ak, self.weight_interface_name)
            return fn(symbol=index_code)

        fhps = _retry(
            _call_fhps,
            retries=self.retries,
            delay=self.retry_delay,
            name=self.dividend_interface_name,
        )
        weight = _retry(
            _call_weight,
            retries=self.retries,
            delay=self.retry_delay,
            name=self.weight_interface_name,
        )
        if fhps is None or fhps.empty or weight is None or weight.empty:
            return pd.DataFrame(
                columns=[
                    "stock_code", "index_code", "ex_date", "dividend_per_share",
                    "weight_pct", "dividend_yield",
                ]
            )

        fhps_code_col = _first_matching_column(fhps, ["代码", "股票代码", "code"])
        div_col = _first_matching_column(fhps, ["每股股利", "每股派息", "现金分红-每股分红"])
        exdate_col = _first_matching_column(fhps, ["除权除息日", "除息日", "股权登记日"])
        price_col = _first_matching_column(fhps, ["最新价", "收盘价", "现价"])
        weight_code_col = _first_matching_column(weight, ["成分券代码", "证券代码", "code"])
        weight_col = _first_matching_column(weight, ["权重", "weight"])

        if None in (fhps_code_col, div_col, exdate_col, weight_code_col, weight_col):
            print(
                f"[akshare] 分红/权重字段无法识别: fhps={list(fhps.columns)}, "
                f"weight={list(weight.columns)}"
            )
            return pd.DataFrame(
                columns=[
                    "stock_code", "index_code", "ex_date", "dividend_per_share",
                    "weight_pct", "dividend_yield",
                ]
            )

        fhps = fhps.rename(columns={
            fhps_code_col: "stock_code", div_col: "dividend_per_share",
            exdate_col: "ex_date",
        })
        weight = weight.rename(columns={weight_code_col: "stock_code", weight_col: "weight_pct"})
        fhps["stock_code"] = fhps["stock_code"].astype(str).str.zfill(6)
        weight["stock_code"] = weight["stock_code"].astype(str).str.zfill(6)
        merged = fhps.merge(weight[["stock_code", "weight_pct"]], on="stock_code", how="inner")
        merged["ex_date"] = pd.to_datetime(merged["ex_date"], errors="coerce")
        merged["dividend_per_share"] = pd.to_numeric(
            merged["dividend_per_share"], errors="coerce"
        ).fillna(0.0)
        merged["weight_pct"] = pd.to_numeric(merged["weight_pct"], errors="coerce").fillna(0.0)
        if price_col is not None:
            price = pd.to_numeric(merged[price_col], errors="coerce")
            merged["dividend_yield"] = np.where(
                price > 0, merged["dividend_per_share"] / price, 0.0
            )
        else:
            merged["dividend_yield"] = 0.0
        merged["index_code"] = index_code
        merged = merged.dropna(subset=["ex_date"])
        out = merged[[
            "stock_code", "index_code", "ex_date", "dividend_per_share",
            "weight_pct", "dividend_yield",
        ]].reset_index(drop=True)
        if not out.empty:
            self.cache.put(key, out)
        return out


# --------------------------------------------------------------------------- #
# CFFEX 官方月度压缩包 + AkShare 现货指数的混合数据源
# --------------------------------------------------------------------------- #

@dataclass
class CffexAkShareAdapter(DataAdapter):
    """原始基差回测的默认真实数据源。

    - 期货：直接下载中金所按月发布的日行情 ZIP，一个月只请求一次。
    - 现货：复用 AkShareAdapter 的中证指数日线接口。
    - 分红：仅为保持 DataAdapter 协议而委托给 AkShare；原始基差模式
      不会调用它。严格分红修正版通过 LocalDividendProvider 读取本地三表。

    AkShare 的 get_futures_daily(CFFEX) 底层也读取同一批中金所
    ZIP，但会按交易日重复下载整月压缩包。本实现按月解压，
    十年回测的网络请求量从约 2500 次降到约 121 次。
    """

    cache_dir: str | Path = ".basis_cache"
    retries: int = 2
    retry_delay: float = 1.0
    base_url: str = "http://www.cffex.com.cn/sj/historysj/{yyyymm}/zip/{yyyymm}.zip"
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.cache = ParquetCache(self.cache_dir)
        self.raw_dir = Path(self.cache_dir) / "cffex_monthly"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.akshare = AkShareAdapter(
            cache_dir=self.cache_dir,
            retries=self.retries,
            retry_delay=self.retry_delay,
        )

    def _download_month(self, yyyymm: str, refresh: bool = False) -> Optional[bytes]:
        path = self.raw_dir / f"{yyyymm}.zip"
        if path.exists() and not refresh:
            try:
                payload = path.read_bytes()
                with zipfile.ZipFile(io.BytesIO(payload)):
                    return payload
            except (OSError, zipfile.BadZipFile):
                path.unlink(missing_ok=True)

        url = self.base_url.format(yyyymm=yyyymm)

        def _call() -> bytes:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            # CFFEX 有时会用 200 返回 HTML 错误页，因此必须验证 ZIP。
            with zipfile.ZipFile(io.BytesIO(response.content)):
                pass
            return response.content

        payload = _retry(
            _call,
            retries=self.retries,
            delay=self.retry_delay,
            name=f"CFFEX monthly {yyyymm}",
        )
        if payload is None:
            return None
        path.write_bytes(payload)
        return payload

    @staticmethod
    def _parse_month(payload: bytes, product: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = sorted(
                n for n in archive.namelist() if re.search(r"\d{8}_1\.csv$", n)
            )
            for name in names:
                match = re.search(r"(\d{8})_1\.csv$", name)
                if match is None:
                    continue
                raw_bytes = archive.read(name)
                text_data = None
                for encoding in ("gb2312", "gbk", "utf-8-sig"):
                    try:
                        text_data = raw_bytes.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                if text_data is None:
                    continue
                daily = pd.read_csv(io.StringIO(text_data))
                symbol_col = _first_matching_column(daily, ["合约代码", "symbol"])
                close_col = _first_matching_column(daily, ["今收盘", "收盘价", "close"])
                settle_col = _first_matching_column(daily, ["今结算", "结算价", "settle"])
                if symbol_col is None or close_col is None:
                    continue
                symbols = daily[symbol_col].astype(str).str.strip().str.upper()
                mask = symbols.str.match(rf"^{re.escape(product)}\d{{4}}$")
                if not mask.any():
                    continue
                frame = pd.DataFrame(
                    {
                        "date": pd.Timestamp(match.group(1)),
                        "contract": symbols.loc[mask],
                        "F": pd.to_numeric(daily.loc[mask, close_col], errors="coerce"),
                        "settle": pd.to_numeric(daily.loc[mask, settle_col], errors="coerce")
                        if settle_col is not None
                        else np.nan,
                    }
                )
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["date", "contract", "F", "settle"])
        return pd.concat(frames, ignore_index=True).dropna(subset=["F"])

    def load_futures(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        key = self.cache.make_key("cffex_futures_trade_calendar_v2", index_code, start_date, end_date)
        if not refresh:
            cached = self.cache.get(key)
            if cached is not None and not cached.empty:
                return cached

        product = INDEX_TO_PRODUCT.get(index_code)
        if product is None:
            raise ValueError(f"未知 index_code: {index_code}")

        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        frames: list[pd.DataFrame] = []
        failed_months: list[str] = []
        for period in pd.period_range(start=start_ts, end=end_ts, freq="M"):
            yyyymm = period.strftime("%Y%m")
            payload = self._download_month(yyyymm, refresh=refresh)
            if payload is None:
                failed_months.append(yyyymm)
                continue
            month_df = self._parse_month(payload, product)
            if not month_df.empty:
                frames.append(month_df)

        if not frames:
            return pd.DataFrame(columns=PANEL_COLUMNS)
        raw = pd.concat(frames, ignore_index=True)
        raw = raw[(raw["date"] >= start_ts) & (raw["date"] <= end_ts)].copy()
        raw = raw.drop_duplicates(["date", "contract"], keep="last")
        # 官方月度文件中的日期本身就是中金所实际交易日历；ZIP 已落本地缓存，
        # 因而无需再依赖另一个网络接口来判断节假日顺延。
        configure_cffex_trade_calendar(raw["date"])

        result = _tag_futures_panel(raw[["date", "contract", "F"]], index_code)
        result.attrs["failed_cffex_months"] = failed_months
        if not result.empty:
            self.cache.put(key, result)
        return result

    def load_index(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        return self.akshare.load_index(index_code, start_date, end_date, refresh=refresh)

    def load_dividends(
        self, index_code: str, year: int, refresh: bool = False
    ) -> pd.DataFrame:
        return self.akshare.load_dividends(index_code, year, refresh=refresh)


# --------------------------------------------------------------------------- #
# LocalFileAdapter：用户自有 CSV / Excel / Parquet 数据切口
# --------------------------------------------------------------------------- #

LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "input" / "local_data"


@dataclass
class LocalFileAdapter(DataAdapter):
    """从本地文件读取期货全合约和现货指数日行情。

    支持 CSV / XLSX / XLS / Parquet，可以通过构造参数显式指定文件，
    也可按 input/local_data/README.md 的默认文件名放置。

    这个 Adapter 只负责价格数据；load_dividends 明确返回空表。
    因此原始基差可直接使用，分红修正仍需另行接入历史权重/分红。
    """

    data_dir: str | Path = LOCAL_DATA_DIR
    futures_path: str | Path | None = None
    index_path: str | Path | None = None
    futures_sheet: str | None = None
    index_sheet: str | None = None
    cache_dir: str | Path = ".basis_cache"  # 保持工厂构造参数兼容，本类不写缓存

    FUTURES_STEMS = ("futures_all_contracts", "futures", "cffex_futures", "价格")
    INDEX_STEMS = ("index_daily", "indices", "spot_indices", "index", "价格")
    EXTENSIONS = (".parquet", ".csv", ".xlsx", ".xls")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.futures_path is not None:
            self.futures_path = Path(self.futures_path)
        if self.index_path is not None:
            self.index_path = Path(self.index_path)

    @staticmethod
    def _parse_dates(series: pd.Series, field_name: str) -> pd.Series:
        # Excel 序列日期（例如 40182 = 2010-01-04）不能交给 pd.to_datetime
        # 直接解析，否则会被当成 1970 年后的纳秒数。
        numeric = pd.to_numeric(series, errors="coerce")
        excel_serial = numeric.between(20_000, 80_000)
        text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        compact = text.str.fullmatch(r"\d{8}")
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if excel_serial.any():
            parsed.loc[excel_serial] = pd.Timestamp("1899-12-30") + pd.to_timedelta(
                numeric.loc[excel_serial], unit="D"
            )
        compact = compact & ~excel_serial
        if compact.any():
            parsed.loc[compact] = pd.to_datetime(
                text.loc[compact], format="%Y%m%d", errors="coerce"
            )
        remaining = ~(compact | excel_serial)
        if remaining.any():
            parsed.loc[remaining] = pd.to_datetime(series.loc[remaining], errors="coerce")
        if parsed.isna().any():
            examples = series.loc[parsed.isna()].head(5).tolist()
            raise ValueError(f"{field_name} 含无法解析的日期，示例: {examples}")
        return parsed.dt.normalize()

    @staticmethod
    def _resolve_sheet(path: Path, requested: str | None, candidates: tuple[str, ...]) -> str | int:
        if requested is not None:
            return requested
        workbook = pd.ExcelFile(path)
        lower_to_original = {str(name).strip().lower(): name for name in workbook.sheet_names}
        for candidate in candidates:
            if candidate.lower() in lower_to_original:
                return lower_to_original[candidate.lower()]
        if len(workbook.sheet_names) == 1:
            return workbook.sheet_names[0]
        raise ValueError(
            f"{path.name} 含多个 sheet，无法自动识别 {candidates}；"
            f"实际 sheet={workbook.sheet_names}"
        )

    @classmethod
    def _read_table(
        cls,
        path: Path,
        requested_sheet: str | None,
        sheet_candidates: tuple[str, ...],
    ) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            last_exc: Exception | None = None
            for encoding in ("utf-8-sig", "gb18030"):
                try:
                    return pd.read_csv(path, encoding=encoding)
                except UnicodeDecodeError as exc:
                    last_exc = exc
            raise ValueError(f"无法读取 CSV 编码: {path}") from last_exc
        if suffix in (".xlsx", ".xls"):
            sheet = cls._resolve_sheet(path, requested_sheet, sheet_candidates)
            raw = pd.read_excel(path, sheet_name=sheet)
            # Wind 导出的宽表前四行是查询元数据，第五行才是机器字段名。
            unnamed_header = raw.columns.size > 0 and all(
                str(col).startswith("Unnamed:") for col in raw.columns
            )
            if ("开始日期" in raw.columns and len(raw) >= 4) or unnamed_header:
                raw = pd.read_excel(path, sheet_name=sheet, header=4)
            return raw
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"不支持的文件格式: {path}")

    def _find_default_path(self, stems: tuple[str, ...]) -> Path:
        matches = [
            self.data_dir / f"{stem}{ext}"
            for stem in stems
            for ext in self.EXTENSIONS
            if (self.data_dir / f"{stem}{ext}").exists()
        ]
        # 也支持一个 market_data.xlsx，期货/指数分别使用不同 sheet。
        for ext in (".xlsx", ".xls"):
            combined = self.data_dir / f"market_data{ext}"
            if combined.exists():
                matches.append(combined)
        unique = list(dict.fromkeys(matches))
        if not unique:
            expected = ", ".join(f"{stems[0]}{ext}" for ext in self.EXTENSIONS)
            raise FileNotFoundError(f"未在 {self.data_dir} 找到数据文件；例如: {expected}")
        if len(unique) > 1:
            raise ValueError(f"同一类数据匹配到多个文件，请删除重复或显式指定: {unique}")
        return unique[0]

    @staticmethod
    def _assert_no_conflicting_duplicates(
        df: pd.DataFrame, keys: list[str], value: str, source: Path
    ) -> pd.DataFrame:
        duplicate_mask = df.duplicated(keys, keep=False)
        if duplicate_mask.any():
            conflicts = (
                df.loc[duplicate_mask]
                .groupby(keys, dropna=False)[value]
                .nunique(dropna=False)
            )
            conflicts = conflicts[conflicts > 1]
            if not conflicts.empty:
                raise ValueError(
                    f"{source.name} 存在同一主键的冲突值，示例: "
                    f"{conflicts.head(5).index.tolist()}"
                )
        return df.drop_duplicates(keys, keep="last")

    def load_futures(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        path = Path(self.futures_path) if self.futures_path else self._find_default_path(self.FUTURES_STEMS)
        raw = self._read_table(
            path,
            self.futures_sheet,
            ("futures", "期货全合约", "股指期货", "期货"),
        )
        date_col = _first_matching_column(raw, ["trade_date", "date", "交易日期", "日期"])
        product = INDEX_TO_PRODUCT.get(index_code)
        if product is None:
            raise ValueError(f"未知 index_code: {index_code}")

        # Wind 宽表：一列日期 + IH00/IH01/IH02/IH03 四列连续合约。
        continuous_cols: dict[int, object] = {}
        for col in raw.columns:
            match = re.fullmatch(rf"{re.escape(product)}0([0-3])\.(?:CFE|CFX|CFFEX)", str(col).upper())
            if match:
                continuous_cols[int(match.group(1))] = col
        if date_col is not None and len(continuous_cols) == 4:
            # Wind 的 00/01/02/03 序号约定 = 当月/次月/季月/下季月，与
            # MONTH_TAG_ORDER 顺序一致。连续合约行情本身不带合约代码，用
            # month_tag_contract_codes() 按交易日纯日历推导出隐含合约代码，
            # 换月时点假定与 CFFEX 到期日规则对齐（第三个周五之后即视为下一
            # 个合约），后续 _tag_futures_panel 会用同一套规则重新计算
            # days_to_expiry，口径统一。
            dates = self._parse_dates(raw[date_col], str(date_col))
            configure_cffex_trade_calendar(dates)
            rows = []
            for row_idx, trade_date in dates.items():
                contract_by_tag = month_tag_contract_codes(trade_date, product)
                for ordinal, tag in enumerate(MONTH_TAG_ORDER):
                    contract = contract_by_tag.get(tag)
                    if contract is None:
                        continue
                    value = pd.to_numeric(raw.at[row_idx, continuous_cols[ordinal]], errors="coerce")
                    if pd.notna(value):
                        rows.append({"date": trade_date, "contract": contract, "F": float(value)})
            normalized = pd.DataFrame(rows, columns=["date", "contract", "F"])
            normalized = self._assert_no_conflicting_duplicates(
                normalized, ["date", "contract"], "F", path
            )
            start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
            normalized = normalized[
                (normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)
            ]
            return _tag_futures_panel(normalized, index_code)

        contract_col = _first_matching_column(raw, ["contract", "symbol", "ts_code", "合约代码", "合约"])
        close_col = "F" if "F" in raw.columns else _first_matching_column(
            raw, ["close", "今收盘", "收盘价", "收盘"]
        )
        if None in (date_col, contract_col, close_col):
            raise ValueError(
                f"{path.name} 期货字段不完整：需要 日期/合约代码/收盘价，"
                f"实际列={list(raw.columns)}"
            )
        normalized = pd.DataFrame(
            {
                "date": self._parse_dates(raw[date_col], str(date_col)),
                "contract": raw[contract_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(r"\.(CFE|CFX|CFFEX)$", "", regex=True),
                "F": pd.to_numeric(raw[close_col], errors="coerce"),
            }
        )
        configure_cffex_trade_calendar(normalized["date"])
        normalized = normalized[
            normalized["contract"].str.match(rf"^{re.escape(product)}\d{{4}}$")
        ].dropna(subset=["F"])
        normalized = self._assert_no_conflicting_duplicates(
            normalized, ["date", "contract"], "F", path
        )
        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        normalized = normalized[
            (normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)
        ]

        return _tag_futures_panel(normalized[["date", "contract", "F"]], index_code)

    def load_index(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        explicit_path = Path(self.index_path) if self.index_path else None
        separate_path = None
        if explicit_path is None:
            for stem in (f"index_{index_code}", index_code):
                found = [self.data_dir / f"{stem}{ext}" for ext in self.EXTENSIONS if (self.data_dir / f"{stem}{ext}").exists()]
                if len(found) > 1:
                    raise ValueError(f"指数 {index_code} 匹配到多个文件: {found}")
                if found:
                    separate_path = found[0]
                    break
        path = explicit_path or separate_path or self._find_default_path(self.INDEX_STEMS)
        raw = self._read_table(
            path,
            self.index_sheet,
            ("indices", "index", "现货指数", "指数日行情", "指数"),
        )
        date_col = _first_matching_column(raw, ["trade_date", "date", "交易日期", "日期"])
        code_col = _first_matching_column(raw, ["index_code", "ts_code", "symbol", "指数代码", "代码"])
        wide_index_col = next(
            (col for col in raw.columns if re.fullmatch(rf"{re.escape(index_code)}(?:\.[A-Z]+)?", str(col).upper())),
            None,
        )
        close_col = wide_index_col or ("S" if "S" in raw.columns else _first_matching_column(
            raw, ["close", "今收盘", "收盘价", "收盘"]
        ))
        if date_col is None or close_col is None:
            raise ValueError(
                f"{path.name} 指数字段不完整：需要 日期/收盘价，"
                f"合并文件还需要指数代码；实际列={list(raw.columns)}"
            )
        if code_col is None and separate_path is None and wide_index_col is None:
            raise ValueError(f"{path.name} 是合并指数文件，但缺少 index_code/指数代码列")
        normalized = pd.DataFrame(
            {
                "date": self._parse_dates(raw[date_col], str(date_col)),
                "index_code": index_code
                if code_col is None
                else raw[code_col]
                .astype(str)
                .str.extract(r"(\d{6})", expand=False)
                .str.zfill(6),
                "S": pd.to_numeric(raw[close_col], errors="coerce"),
            }
        )
        normalized = normalized[
            normalized["index_code"].eq(index_code)
        ].dropna(subset=["S"])
        normalized = self._assert_no_conflicting_duplicates(
            normalized, ["date", "index_code"], "S", path
        )
        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        return normalized[
            (normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)
        ].sort_values("date").reset_index(drop=True)

    def load_dividends(
        self, index_code: str, year: int, refresh: bool = False
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "stock_code", "index_code", "ex_date", "dividend_per_share",
                "weight_pct", "dividend_yield",
            ]
        )


# --------------------------------------------------------------------------- #
# 本地 point-in-time 分红数据（独立于价格 DataAdapter）
# --------------------------------------------------------------------------- #

class DividendAdapter(abc.ABC):
    """Point-in-time 分红数据接口；下游不依赖具体供应商。"""

    @abc.abstractmethod
    def validate_access(self) -> tuple[bool, str]:
        """校验生成 upper-bound 所需的基础字段；announced 资格另行判断。"""

    @abc.abstractmethod
    def load_dataset(
        self, index_code: str, start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """返回 events / weights / prices 三张统一 schema 长表。"""


@dataclass
class LocalDividendAdapter(DividendAdapter):
    """读取本地分红事件、历史指数权重和股票价格三张长表。"""

    data_dir: str | Path = Path(__file__).resolve().parent.parent / "input" / "local_dividend"
    EXTENSIONS = (".parquet", ".csv", ".xlsx", ".xls")
    FILE_STEMS = {
        "events": "dividend_events",
        "weights": "index_weights",
        "prices": "stock_prices",
    }

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)

    def _path(self, kind: str) -> Path:
        stem = self.FILE_STEMS[kind]
        matches = [self.data_dir / f"{stem}{ext}" for ext in self.EXTENSIONS if (self.data_dir / f"{stem}{ext}").exists()]
        if len(matches) != 1:
            raise FileNotFoundError(f"{kind} 需要且只能有一个文件: {stem}{{{','.join(self.EXTENSIONS)}}}")
        return matches[0]

    def _read(self, kind: str) -> pd.DataFrame:
        return LocalFileAdapter._read_table(self._path(kind), None, (kind, self.FILE_STEMS[kind]))

    @staticmethod
    def _column(frame: pd.DataFrame, candidates: list[str], label: str):
        col = _first_matching_column(frame, candidates)
        if col is None:
            raise ValueError(f"缺少字段 {label}；实际列={list(frame.columns)}")
        return col

    def validate_access(self) -> tuple[bool, str]:
        try:
            self.load_dataset("000300", "1900-01-01", "2100-12-31")
            return True, "本地分红三表字段验收通过"
        except Exception as exc:
            return False, f"本地分红数据不可用: {type(exc).__name__}: {exc}"

    def load_dataset(self, index_code: str, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        events_raw, weights_raw, prices_raw = self._read("events"), self._read("weights"), self._read("prices")
        ann_col = _first_matching_column(events_raw, ["ann_date", "公告日", "预案公告日"])
        events = pd.DataFrame({
            "stock_code": events_raw[self._column(events_raw, ["stock_code", "ts_code", "股票代码", "证券代码"], "stock_code")].astype(str),
            "index_code": events_raw[self._column(events_raw, ["index_code", "指数代码"], "index_code")].astype(str).str.extract(r"(\d{6})", expand=False),
            # ann_date 允许整列缺失，以便仍可生成明确标脏的 upper-bound；
            # announced 的资格由调用方按非空完整性单独判断。
            "ann_date": (
                LocalFileAdapter._parse_dates(events_raw[ann_col], "ann_date")
                if ann_col is not None else pd.Series(pd.NaT, index=events_raw.index)
            ),
            "ex_date": LocalFileAdapter._parse_dates(events_raw[self._column(events_raw, ["ex_date", "除权除息日", "除息日"], "ex_date")], "ex_date"),
            "cash_div": pd.to_numeric(events_raw[self._column(events_raw, ["cash_div", "每股现金分红", "每股派息"], "cash_div")], errors="coerce"),
        })
        end_col = _first_matching_column(events_raw, ["end_date", "报告期", "分红年度"])
        if end_col is not None:
            events["end_date"] = pd.to_datetime(events_raw[end_col], errors="coerce")
        weights = pd.DataFrame({
            "stock_code": weights_raw[self._column(weights_raw, ["stock_code", "con_code", "ts_code", "股票代码", "成分券代码"], "stock_code")].astype(str),
            "index_code": weights_raw[self._column(weights_raw, ["index_code", "指数代码"], "index_code")].astype(str).str.extract(r"(\d{6})", expand=False),
            "trade_date": LocalFileAdapter._parse_dates(weights_raw[self._column(weights_raw, ["trade_date", "date", "权重日期", "日期"], "trade_date")], "trade_date"),
            "weight_pct": pd.to_numeric(weights_raw[self._column(weights_raw, ["weight_pct", "weight", "权重"], "weight_pct")], errors="coerce"),
        })
        prices = pd.DataFrame({
            "stock_code": prices_raw[self._column(prices_raw, ["stock_code", "ts_code", "股票代码", "证券代码"], "stock_code")].astype(str),
            "trade_date": LocalFileAdapter._parse_dates(prices_raw[self._column(prices_raw, ["trade_date", "date", "日期"], "trade_date")], "trade_date"),
            "close": pd.to_numeric(prices_raw[self._column(prices_raw, ["close", "收盘价", "收盘"], "close")], errors="coerce"),
        })
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        events = events[events["index_code"].eq(index_code)].dropna(subset=["ex_date", "cash_div"])
        weights = weights[
            weights["index_code"].eq(index_code)
            & (weights["trade_date"] <= end)
        ].dropna(subset=["weight_pct"])
        stocks = set(weights["stock_code"])
        prices = prices[
            prices["stock_code"].isin(stocks)
            & (prices["trade_date"] <= end)
        ].dropna(subset=["close"])
        if events.empty or weights.empty or prices.empty:
            raise ValueError(f"指数 {index_code} 在所需区间缺少 events/weights/prices")
        return {"events": events.reset_index(drop=True), "weights": weights.reset_index(drop=True), "prices": prices.reset_index(drop=True)}


# 兼容早期调用名称；新业务代码统一经 get_dividend_adapter() 获取。
LocalDividendProvider = LocalDividendAdapter

_DIVIDEND_ADAPTER_REGISTRY: dict[str, type[DividendAdapter]] = {
    "local": LocalDividendAdapter,
}


def get_dividend_adapter(name: str, **cfg) -> DividendAdapter:
    """分红数据源的唯一切换点；当前按要求只保留本地文件实现。"""
    try:
        adapter_cls = _DIVIDEND_ADAPTER_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"未知分红数据源: {name!r}，可选: {sorted(_DIVIDEND_ADAPTER_REGISTRY)}"
        ) from exc
    return adapter_cls(**cfg)


# --------------------------------------------------------------------------- #
# WindAdapter：真实数据源桩实现（预留给有 Wind 授权用户填入）
# --------------------------------------------------------------------------- #

@dataclass
class WindAdapter(DataAdapter):
    """Wind 数据源桩实现。

    本类是空壳：三个接口都只做参数校验然后 raise NotImplementedError，具体取数
    逻辑留给有 Wind (万得) 终端授权、能跑通 WindPy 的用户按下面的 TODO 注释填入。
    这里只负责规定"填完之后必须产出和 AkShareAdapter / MockAdapter 完全一致的
    面板 schema"这件事，好让上层四层流水线（factors/regression/strategy/forecast/
    run）在切换数据源时不用改一行业务逻辑。

    切换方式：不要在业务代码里直接 new 这个类，统一用 get_adapter('wind', **cfg)。
    """

    cache_dir: str | Path = ".basis_cache"

    def __post_init__(self) -> None:
        self.cache = ParquetCache(self.cache_dir)

    def load_futures(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        """加载全合约期货日行情，需返回 PANEL_COLUMNS 中除 S 外的列。

        TODO（需要 Wind 授权用户实现）：
            1. `from WindPy import w; w.start()`（放在方法内部，禁止模块级 import，
               保持"只有 data.py 允许触碰具体数据源 SDK"的纪律）。
            2. 用 `w.wset("sectorconstituent", ...)` 或 `w.wset("futurecc", ...)`
               取 index_code 对应品种（IH/IF/IC/IM）在 [start_date, end_date] 内
               的全合约列表。
            3. 对每个合约用 `w.wsd(code, "close", start_date, end_date)` 取日收盘价。
            4. 用 data.assign_month_tags() / data.days_to_expiry() 补齐 month_tag
               与 days_to_expiry，产出与 AkShareAdapter.load_futures 完全一致的
               DataFrame（列: date, contract, index_code, month_tag, F, S(NaN),
               days_to_expiry），S 留空由 merge_panel() 与 load_index 结果合并补齐。
        """
        raise NotImplementedError(
            "WindAdapter.load_futures 是桩实现，请接入 WindPy (w.wsd/w.wset) 后移除本异常。"
        )

    def load_index(
        self, index_code: str, start_date: str, end_date: str, refresh: bool = False
    ) -> pd.DataFrame:
        """加载现货指数日收盘价，需返回列: date, index_code, S。

        TODO（需要 Wind 授权用户实现）：用 `w.wsd(index_wind_code, "close",
        start_date, end_date)` 取指数收盘价，index_code(000300等) 需要先映射成
        Wind 代码（如 "000300.SH"）。
        """
        raise NotImplementedError(
            "WindAdapter.load_index 是桩实现，请接入 WindPy (w.wsd) 后移除本异常。"
        )

    def load_dividends(
        self, index_code: str, year: int, refresh: bool = False
    ) -> pd.DataFrame:
        """加载成分股分红预测面板，需返回列: stock_code, index_code, ex_date,
        dividend_per_share, weight_pct, dividend_yield。

        TODO（需要 Wind 授权用户实现）：用 `w.wset("sectorconstituent", ...)` 取
        成分股权重，用 `w.wsd(stock_code, "div_cashbeforetax", ...)` 或
        `w.wset("dividend", ...)` 取每股股利与除权除息日，参照
        AkShareAdapter.load_dividends 的合并/折算逻辑产出同样的 schema。
        """
        raise NotImplementedError(
            "WindAdapter.load_dividends 是桩实现，请接入 WindPy (w.wset/w.wsd) 后移除本异常。"
        )


# --------------------------------------------------------------------------- #
# 数据源切换：统一工厂函数，业务代码只认 DataAdapter 接口
# --------------------------------------------------------------------------- #

#: 数据源名称 -> 具体实现类。切换数据源只改这里传给 get_adapter 的 name/cfg，
#: 不需要改 factors/regression/strategy/forecast/run 里的任何一行代码。
_ADAPTER_REGISTRY: dict[str, type[DataAdapter]] = {
    "mock": MockAdapter,
    "akshare": AkShareAdapter,
    "cffex_akshare": CffexAkShareAdapter,
    "local": LocalFileAdapter,
    "wind": WindAdapter,
}


def get_adapter(name: str, **cfg) -> DataAdapter:
    """数据源工厂：按名字返回对应的 DataAdapter 实例。

    这是切换数据源的唯一入口 —— 以后从 akshare 换到 Wind（或换到任何新数据源），
    只需要改调用方传给这个函数的 name/cfg 参数（或配置文件里的一个字段），
    不需要改 data.py 以外任何模块的代码，因为上层永远只依赖 DataAdapter 这个
    抽象接口和标准面板 schema。

    Args:
        name: 'mock' / 'akshare' / 'wind'（大小写不敏感）。
        **cfg: 透传给具体 Adapter 构造函数的参数，如 cache_dir、retries 等。

    Returns:
        对应的 DataAdapter 实例。

    Raises:
        ValueError: name 不在已注册的数据源列表里。
    """
    key = name.strip().lower()
    adapter_cls = _ADAPTER_REGISTRY.get(key)
    if adapter_cls is None:
        raise ValueError(
            f"未知数据源: {name!r}，可选: {sorted(_ADAPTER_REGISTRY.keys())}"
        )
    return adapter_cls(**cfg)


# --------------------------------------------------------------------------- #
# 组装工具：合并期货面板 + 现货 S
# --------------------------------------------------------------------------- #

def merge_panel(futures_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """把 load_futures 输出（S 可能为空）与 load_index 输出按 date+index_code 合并。"""
    if futures_df.empty:
        return futures_df
    merged = futures_df.drop(columns=["S"], errors="ignore").merge(
        index_df[["date", "index_code", "S"]], on=["date", "index_code"], how="left"
    )
    merged = merged.dropna(subset=["S", "F"]).reset_index(drop=True)
    return merged[PANEL_COLUMNS]


def load_panel_with_fallback(
    primary: DataAdapter,
    fallback: Optional[DataAdapter],
    index_code: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """优先用 primary（AkShareAdapter/WindAdapter 等任意 DataAdapter 实现）取数，
    失败/为空/未实现（如 WindAdapter 桩实现直接 raise NotImplementedError）则回退
    fallback（一般是 MockAdapter）。primary 具体是哪个数据源由调用方通过
    data.get_adapter(name) 决定，本函数不关心，只依赖 DataAdapter 抽象接口。
    """
    source_name = type(primary).__name__
    try:
        futures = primary.load_futures(index_code, start_date, end_date, refresh=refresh)
        idx = primary.load_index(index_code, start_date, end_date, refresh=refresh)
        panel = merge_panel(futures, idx)
        if not panel.empty:
            panel.attrs["data_source"] = source_name
            return panel
        print(f"[data] {source_name} 数据为空({index_code})。")
    except Exception as exc:
        print(f"[data] {source_name} 取数异常({index_code}): {exc!r}。")

    if fallback is None:
        raise RuntimeError(
            f"{source_name} 未取到 {index_code} 的真实数据；"
            "为防止把 MockAdapter 模拟数据误当成十年历史回测，本次已停止。"
        )
    panel = fallback.generate_mock_panel(index_code, start_date, end_date)
    panel.attrs["data_source"] = type(fallback).__name__
    print(f"[data] 明确允许模拟回测，{index_code} 使用 MockAdapter。")
    return panel
