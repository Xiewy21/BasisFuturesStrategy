"""config.py —— 十年期理论回测的运行配置。

方法论性质：无（纯配置），但明确声明这一轮回测的口径：
    - "理论回测" = 不对照任何研究报告的历史数字，只用统一、可复现的规则和参数，
      观察策略/因子在长历史窗口（2015 至今）下的统计特征是否稳定。
    - "无摩擦" = 交易成本 cost 默认设为 0，回测得到的是策略的理论上限，不代表
      扣除滑点/手续费/冲击成本后的可实现收益（cost 参数保留，以后要加摩擦
      成本只需要改这一个配置项，不需要动 run.py / strategy.py 的逻辑）。

run.py 只从这个文件读取参数；换指数、换区间、换 N、换数据源都只改这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# 回测区间
# --------------------------------------------------------------------------- #

def _ten_years_ago(today: date) -> date:
    """返回真正的滚动十年起点，并处理 2 月 29 日。"""
    try:
        return today.replace(year=today.year - 10)
    except ValueError:
        return today.replace(year=today.year - 10, day=28)


_TODAY = date.today()
START_DATE: str = _ten_years_ago(_TODAY).isoformat()
END_DATE: str = _TODAY.isoformat()


@dataclass(frozen=True)
class IndexConfig:
    """单个指数/期货品种在本轮十年回测里的配置。"""

    index_code: str  # 现货指数代码
    product: str  # 期货品种代码（IH/IF/IC/IM）
    name: str  # 中文名，仅用于打印/展示
    data_start: str  # 该指数在本轮回测里实际使用的起始日（可能晚于全局 START_DATE）
    short_history_note: Optional[str] = None  # 历史样本明显偏短时的提示文案


#: 四个指数的配置。IF/IH/IC 使用滚动十年起点；IM（中证1000
#: 股指期货，2022-07-22 才上市）历史样本显著更短，
#: 单独标注，解读稳健性结论时需要格外谨慎。
INDEX_CONFIGS: list[IndexConfig] = [
    IndexConfig(index_code="000016", product="IH", name="上证50", data_start=START_DATE),
    IndexConfig(index_code="000300", product="IF", name="沪深300", data_start=START_DATE),
    IndexConfig(index_code="000905", product="IC", name="中证500", data_start=START_DATE),
    IndexConfig(
        index_code="000852",
        product="IM",
        name="中证1000",
        data_start="2022-07-22",
        short_history_note="IM 合约 2022-07-22 才上市，样本量明显少于其他三个品种"
        "（不足其他品种的三分之一），任何稳健性/显著性结论都需要打折看待。",
    ),
]

# --------------------------------------------------------------------------- #
# 数据源
# --------------------------------------------------------------------------- #

#: 主数据源；'wind' 目前是桩实现（见 data.WindAdapter），会被自动检测并回退。
DATA_SOURCE_PRIMARY: str = "local"
#: 主数据源不可用时的第二级数据源。
DATA_SOURCE_FALLBACK: str = "akshare"
#: 第二级也失败时的最终兜底，由 data.load_panel_with_fallback 负责。
DATA_SOURCE_LAST_RESORT: str = "mock"
# 真实回测默认不允许静默使用模拟数据。只有做流水线演示/单测时才开启。
ALLOW_MOCK_BACKTEST: bool = False

# --------------------------------------------------------------------------- #
# 分红
# --------------------------------------------------------------------------- #

#: 本轮理论回测的默认口径是剔分红（assume_zero_dividend=False）。如果某个指数
#: 在整个回测区间都拿不到任何一年的分红/权重数据，run.py 会针对该指数单独回退
#: 到 True，并打印"分红季基差未剔除"告警，不影响其他指数继续跑。
#: 当前默认运行“原始基差版”：分红点位固定为 0。待本地 point-in-time
#: 分红三表接入后，另行运行 dividend_adjusted 目录，
#: 不覆盖这一组原始结果。
ASSUME_ZERO_DIVIDEND_DEFAULT: bool = True

# --------------------------------------------------------------------------- #
# 收敛窗口 / 回归 / 择时参数
# --------------------------------------------------------------------------- #

#: 主 N（某券商口径）。
N_HORIZON: int = 8
#: 稳健性对照扫描的额外 N 值（不含主 N，run.py 会自动去重合并）。
N_ROBUSTNESS: tuple[int, ...] = (5, 10, 20)

#: walk-forward 滚动窗口（交易日）。
ROLLING_WINDOW: int = 252
#: 每隔多少个交易日重新估计一次系数（对应 regression.rolling_ols 的 step 参数）。
REFIT_EVERY: int = 5
#: 滚动窗口内最少观测数。
MIN_OBS: int = 60

#: 择时信号滚动分位数窗口 / 阈值。
TIMING_WINDOW: int = 60
TIMING_PCT: float = 0.2

#: 固定方向比较使用约三年历史的因果分位阈值；至少积累一年历史才出信号。
SIGNAL_DIRECTION_WINDOW: int = 252 * 3
SIGNAL_DIRECTION_MIN_PERIODS: int = 252
SIGNAL_DIRECTION_TAIL_PCT: float = 0.10

#: 动态选向只使用这一套预注册参数，不扫描 N、分位阈值或门槛。
WALKFORWARD_TRAIN_YEARS: int = 3
WALKFORWARD_TEST_MONTHS: int = 3
#: 三个非 flat 候选的一侧 5% Bonferroni 门槛约为 2.13。
WALKFORWARD_HAC_TSTAT_MIN: float = 2.13
WALKFORWARD_SHARPE_NW_MIN: float = 0.50

#: 交易成本（理论口径 = 0），保留参数接口，以后要算摩擦成本只改这一个数字。
COST: float = 0.0

#: 敏感度表的情景假设（未来 N 日现货预期收益率）。
SENSITIVITY_EXPECTED_RETURNS: tuple[float, ...] = (-0.04, -0.02, 0.0, 0.02, 0.04)

# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "output" / "raw_basis"
CACHE_DIR: Path = Path(__file__).resolve().parent.parent / ".basis_cache"


@dataclass(frozen=True)
class BacktestConfig:
    """打包成一个对象，方便 run.py 整体传递；字段默认值取自上面的模块级常量。"""

    start_date: str = START_DATE
    end_date: str = END_DATE
    index_configs: tuple[IndexConfig, ...] = field(default_factory=lambda: tuple(INDEX_CONFIGS))
    data_source_primary: str = DATA_SOURCE_PRIMARY
    data_source_fallback: str = DATA_SOURCE_FALLBACK
    allow_mock_backtest: bool = ALLOW_MOCK_BACKTEST
    assume_zero_dividend_default: bool = ASSUME_ZERO_DIVIDEND_DEFAULT
    n_horizon: int = N_HORIZON
    n_robustness: tuple[int, ...] = N_ROBUSTNESS
    rolling_window: int = ROLLING_WINDOW
    refit_every: int = REFIT_EVERY
    min_obs: int = MIN_OBS
    timing_window: int = TIMING_WINDOW
    timing_pct: float = TIMING_PCT
    signal_direction_window: int = SIGNAL_DIRECTION_WINDOW
    signal_direction_min_periods: int = SIGNAL_DIRECTION_MIN_PERIODS
    signal_direction_tail_pct: float = SIGNAL_DIRECTION_TAIL_PCT
    walkforward_train_years: int = WALKFORWARD_TRAIN_YEARS
    walkforward_test_months: int = WALKFORWARD_TEST_MONTHS
    walkforward_hac_tstat_min: float = WALKFORWARD_HAC_TSTAT_MIN
    walkforward_sharpe_nw_min: float = WALKFORWARD_SHARPE_NW_MIN
    cost: float = COST
    sensitivity_expected_returns: tuple[float, ...] = SENSITIVITY_EXPECTED_RETURNS
    output_dir: Path = OUTPUT_DIR
    cache_dir: Path = CACHE_DIR

    @property
    def all_n_values(self) -> list[int]:
        """主 N 与稳健性扫描 N 去重合并后的有序列表，主 N 排在最前面。"""
        seen = [self.n_horizon]
        for n in self.n_robustness:
            if n not in seen:
                seen.append(n)
        return seen


DEFAULT_CONFIG = BacktestConfig()


# --------------------------------------------------------------------------- #
# 基差复盘分析模块（term_structure / basis_center / convergence_prob /
# neutral_hedge_timing）参数。四个模块共用同一份配置，任何口径调整只改这里，
# 不动各模块内部逻辑。
# --------------------------------------------------------------------------- #

#: 复盘输出统一落到 output/<variant>/review/ 下。
REVIEW_OUTPUT_SUBDIR: str = "review"

#: 阶段一：常数期限网格（单位：自然日），端点外不外推。
TERM_GRID_DAYS: tuple[int, ...] = (30, 60, 90, 180)
#: 常数期限插值方法："linear" 或 "cubic"。
TERM_INTERP_METHOD: str = "linear"
#: PCA 保留的主成分个数。
TERM_PCA_N_COMPONENTS: int = 3

#: 阶段二：中枢滚动窗口（交易日），主口径 250 日，另跑 500 日对照。
BASIS_CENTER_WINDOW: int = 250
BASIS_CENTER_WINDOW_ALT: int = 500
#: 滚动分位带。
BASIS_CENTER_QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75)
#: 中枢窗口内最少观测数（不足则该日中枢/std 为 NaN）。
BASIS_CENTER_MIN_PERIODS: int = 60
#: 统计口径异常阈值：滚动分位 <5%/>95%，等价于双尾各 5%。
ANOMALY_TAIL_PCT: float = 0.05
#: |z| 异常阈值。
ANOMALY_Z_THRESHOLD: float = 2.0
#: 持续性口径：连续处于极端分位的最少交易日数。
ANOMALY_PERSISTENCE_M: int = 3

#: 阶段三：收敛事件的未来窗口（交易日），主 N=8，另跑 20/60 对照。
CONVERGENCE_N_HORIZONS: tuple[int, ...] = (8, 20, 60)
#: 收敛判定阈值：|f1-中枢| < 0.5*滚动std，或相对初始偏离收敛掉的比例。
CONVERGENCE_STD_THRESHOLD: float = 0.5
CONVERGENCE_DECAY_THRESHOLD: float = 0.5
#: 到期效应分层：主结果只用剩余交易日 > 20 的样本。
CONVERGENCE_MIN_DTE: int = 20
#: z 分桶边界（不含 ±inf，函数内部会补上）。
CONVERGENCE_Z_BUCKET_EDGES: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
#: block bootstrap 重复次数与置信区间。
CONVERGENCE_BOOTSTRAP_REPS: int = 500
CONVERGENCE_BOOTSTRAP_CI: tuple[float, float] = (0.025, 0.975)
CONVERGENCE_BOOTSTRAP_SEED: int = 42

#: 阶段四：中性策略择时的 purge/embargo 天数，沿用主 N。
HEDGE_TIMING_PURGE_DAYS: int = N_HORIZON
HEDGE_TIMING_EMBARGO_DAYS: int = N_HORIZON
#: walk-forward 训练/测试窗口，沿用动态选向同一套口径。
HEDGE_TIMING_TRAIN_YEARS: int = WALKFORWARD_TRAIN_YEARS
HEDGE_TIMING_TEST_MONTHS: int = WALKFORWARD_TEST_MONTHS
#: 富贵度->敞口映射中，收敛概率对基础富贵度的最大加权修正幅度。
HEDGE_TIMING_TILT_WEIGHT: float = 0.4
#: 富贵度 richness 的 z 值裁剪范围（超出裁剪到边界）。
HEDGE_TIMING_RICHNESS_Z_CLIP: float = 3.0


@dataclass(frozen=True)
class ReviewConfig:
    """基差复盘四个模块共用的参数集合，字段默认值取自上面的模块级常量。"""

    output_subdir: str = REVIEW_OUTPUT_SUBDIR
    rolling_window: int = ROLLING_WINDOW
    min_obs: int = MIN_OBS
    refit_every: int = REFIT_EVERY
    n_horizon: int = N_HORIZON

    term_grid_days: tuple[int, ...] = TERM_GRID_DAYS
    term_interp_method: str = TERM_INTERP_METHOD
    term_pca_n_components: int = TERM_PCA_N_COMPONENTS

    basis_center_window: int = BASIS_CENTER_WINDOW
    basis_center_window_alt: int = BASIS_CENTER_WINDOW_ALT
    basis_center_quantiles: tuple[float, ...] = BASIS_CENTER_QUANTILES
    basis_center_min_periods: int = BASIS_CENTER_MIN_PERIODS
    anomaly_tail_pct: float = ANOMALY_TAIL_PCT
    anomaly_z_threshold: float = ANOMALY_Z_THRESHOLD
    anomaly_persistence_m: int = ANOMALY_PERSISTENCE_M

    convergence_n_horizons: tuple[int, ...] = CONVERGENCE_N_HORIZONS
    convergence_std_threshold: float = CONVERGENCE_STD_THRESHOLD
    convergence_decay_threshold: float = CONVERGENCE_DECAY_THRESHOLD
    convergence_min_dte: int = CONVERGENCE_MIN_DTE
    convergence_z_bucket_edges: tuple[float, ...] = CONVERGENCE_Z_BUCKET_EDGES
    convergence_bootstrap_reps: int = CONVERGENCE_BOOTSTRAP_REPS
    convergence_bootstrap_ci: tuple[float, float] = CONVERGENCE_BOOTSTRAP_CI
    convergence_bootstrap_seed: int = CONVERGENCE_BOOTSTRAP_SEED

    hedge_timing_purge_days: int = HEDGE_TIMING_PURGE_DAYS
    hedge_timing_embargo_days: int = HEDGE_TIMING_EMBARGO_DAYS
    hedge_timing_train_years: int = HEDGE_TIMING_TRAIN_YEARS
    hedge_timing_test_months: int = HEDGE_TIMING_TEST_MONTHS
    hedge_timing_tilt_weight: float = HEDGE_TIMING_TILT_WEIGHT
    hedge_timing_richness_z_clip: float = HEDGE_TIMING_RICHNESS_Z_CLIP


DEFAULT_REVIEW_CONFIG = ReviewConfig()
