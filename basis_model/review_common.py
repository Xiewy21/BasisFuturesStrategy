"""review_common.py —— 基差复盘四个模块（term_structure / basis_center /
convergence_prob / neutral_hedge_timing）共用的 I/O 工具。

方法论性质：无（纯工程），只负责发现/加载已有的 factor_panel.csv，以及统一
复盘结果的落盘位置 output/<variant>/review/。不做任何统计计算。

【与既有流水线的关系】
    本模块只读 run.py / run_dividend_variants.py 已经写好的 factor_panel.csv，
    不重新跑 data -> dividend -> factors 流水线，因此复盘四个模块任何一个都
    不依赖彼此先跑完（各自 main() 都从磁盘现有的 factor_panel.csv 出发），
    但都要求至少有一个 variant 已经跑过 run.py / run_dividend_variants.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OUTPUT_ROOT: Path = Path(__file__).resolve().parent.parent / "output"


@dataclass(frozen=True)
class VariantIndex:
    """一个 (basis_variant, index_code) 组合的定位信息。"""

    variant: str
    index_code: str
    name: str
    factor_panel_path: Path


def discover_variant_indices(output_root: Path = OUTPUT_ROOT) -> list[VariantIndex]:
    """扫描 output_root 下所有含 summary.csv 的 variant 目录，列出其可用指数。

    只认 summary.csv 里登记过的 (index_code, name)，且要求对应的
    factor_panel.csv 确实存在；两者缺一都跳过该指数，不抛异常，方便调用方
    在数据尚未跑全时也能先处理已经就绪的部分。
    """
    results: list[VariantIndex] = []
    if not output_root.exists():
        return results
    for variant_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        summary_path = variant_dir / "summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path, dtype={"index_code": str})
        if "index_code" not in summary.columns or "name" not in summary.columns:
            continue
        for row in summary.itertuples(index=False):
            code = str(row.index_code).zfill(6)
            panel_path = variant_dir / "indices" / f"{code}_{row.name}" / "factor_panel.csv"
            if panel_path.exists():
                results.append(VariantIndex(variant_dir.name, code, str(row.name), panel_path))
    return results


def load_factor_panel(path: Path) -> pd.DataFrame:
    """按项目统一口径读回 factor_panel.csv（date 解析成 Timestamp，index_code 保留前导零）。"""
    return pd.read_csv(path, parse_dates=["date"], dtype={"index_code": str})


def review_output_dir(
    output_root: Path, variant: str, review_subdir: str = "review"
) -> Path:
    """复盘结果统一目录：output/<variant>/review/。调用方负责后续文件名。"""
    out_dir = output_root / variant / review_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def assert_no_future_leakage(reference_dates: pd.Series, source_dates: pd.Series) -> None:
    """单测/自检用：确认 source_dates 中不存在任何晚于 reference_dates 对应截止日的观测。

    用于各模块的“因果滚动”单测：调用方传入某个截止日 cutoff 对应的
    reference_dates（通常是重复同一个 cutoff 的 Series），以及该截止日实际
    用到的历史观测日期 source_dates，若 source_dates 里出现晚于 cutoff 的日期
    则抛异常。纯断言函数，不参与正常计算路径。
    """
    cutoff = pd.Timestamp(reference_dates.max())
    leaked = pd.to_datetime(source_dates)[pd.to_datetime(source_dates) > cutoff]
    if len(leaked):
        raise AssertionError(
            f"防前视违规：截止日 {cutoff.date()} 之后的观测 {leaked.tolist()[:5]} "
            "被用于本应因果滚动的统计量。"
        )
