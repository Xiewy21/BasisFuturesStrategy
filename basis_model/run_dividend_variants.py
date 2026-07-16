"""从本地 point-in-time 文件生成 announced 与含前视 upper-bound 口径。"""

from __future__ import annotations

from dataclasses import replace
import logging

import pandas as pd

from basis_model import config as config_mod
from basis_model import data as data_mod
from basis_model import dividend as dividend_mod
from basis_model import run as run_mod

logger = logging.getLogger("basis_model.run_dividend_variants")

VARIANT_DIRS = {
    "announced_dividend": "dividend_adjusted",
    "realized_dividend_upper_bound": "realized_dividend_upper_bound",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base = config_mod.DEFAULT_CONFIG
    # 三口径必须沿用现有 raw 的行情数据源，避免把数据源差异误当分红差异。
    raw_summary_path = base.output_dir / "summary.csv"
    if raw_summary_path.exists():
        raw_sources = pd.read_csv(raw_summary_path).get("data_source", pd.Series(dtype=str))
        if not raw_sources.empty and raw_sources.eq("LocalFileAdapter").all():
            base = replace(base, data_source_primary="local", data_source_fallback="local")
    provider = data_mod.get_dividend_adapter("local")
    available, message = provider.validate_access()
    if not available:
        status_dir = base.output_dir.parent / "dividend_adjusted"
        status_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "basis_variant": "announced_dividend",
            "available": False,
            "formal_backtest_eligible": False,
            "reason": message,
            "dividend_source": "local point-in-time files",
        }]).to_csv(status_dir / "run_manifest.csv", index=False)
        logger.warning("%s；整体跳过 announced/upper_bound，不生成伪修正结果。", message)
        return

    adapter = run_mod.resolve_primary_adapter(base)
    prepared = []
    try:
        for index_cfg in base.index_configs:
            loaded = run_mod._load_data_layer(
                index_cfg, adapter, None, index_cfg.data_start, base.end_date
            )
            if loaded is None:
                raise RuntimeError(f"{index_cfg.name} 原始价格面板为空")
            panel, source = loaded
            dataset = provider.load_dataset(index_cfg.index_code, index_cfg.data_start, base.end_date)
            prepared.append((index_cfg, panel, source, dataset))
    except Exception as exc:
        logger.warning("四个指数的本地分红三表未全部通过验收: %s；整体跳过，不生成部分结果。", exc)
        return

    announced_eligible = all(
        not dataset["events"].empty and dataset["events"]["ann_date"].notna().all()
        for _, _, _, dataset in prepared
    )
    active_variants = ["realized_dividend_upper_bound"]
    if announced_eligible:
        active_variants.insert(0, "announced_dividend")
    else:
        status_dir = base.output_dir.parent / "dividend_adjusted"
        status_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "basis_variant": "announced_dividend",
            "available": False,
            "formal_backtest_eligible": False,
            "reason": "ann_date 缺失或不完整",
            "dividend_source": "local point-in-time files",
        }]).to_csv(status_dir / "run_manifest.csv", index=False)
        logger.warning(
            "ann_date 缺失或不完整：跳过 announced；仍生成含前视、仅作上界的 upper-bound。"
        )

    summaries: dict[str, list[dict]] = {variant: [] for variant in active_variants}
    for index_cfg, panel, source, dataset in prepared:
        expected_keys = panel[["date", "contract"]].reset_index(drop=True)

        for variant in active_variants:
            dirname = VARIANT_DIRS[variant]
            cfg = replace(base, output_dir=base.output_dir.parent / dirname)
            adjusted = dividend_mod.dividend_adjust_panel_point_in_time(
                panel, dataset["events"], dataset["weights"], dataset["prices"], variant
            )
            if len(adjusted) != len(panel) or not adjusted[["date", "contract"]].reset_index(drop=True).equals(expected_keys):
                raise AssertionError(f"{index_cfg.name} {variant} 与 raw 面板行集合不一致")
            robustness, factors, coefs = run_mod._factor_regression_layer(adjusted, index_cfg, cfg)
            if factors is None or coefs is None:
                raise RuntimeError(f"{index_cfg.name} {variant} 未产出主回归")
            daily = run_mod._forward_fill_daily_coefs(factors, coefs)
            backtest, ladder = run_mod._strategy_layer(factors, coefs, daily, index_cfg, cfg)
            sensitivity = run_mod._forecast_layer(factors, coefs, index_cfg, cfg)
            run_mod._write_outputs(cfg, index_cfg, factors, coefs, backtest, ladder, sensitivity, robustness)
            row = run_mod._build_summary_row(
                index_cfg, index_cfg.data_start, factors, coefs, backtest, ladder,
                source, assume_zero_dividend=False,
            )
            row["basis_variant"] = variant
            row["contains_lookahead"] = variant == "realized_dividend_upper_bound"
            summaries[variant].append(row)

    for variant in active_variants:
        dirname = VARIANT_DIRS[variant]
        target_dir = base.output_dir.parent / dirname
        pd.DataFrame(summaries[variant]).to_csv(target_dir / "summary.csv", index=False)
        pd.DataFrame([{
            "basis_variant": variant,
            "available": True,
            "contains_lookahead": variant == "realized_dividend_upper_bound",
            "formal_backtest_eligible": variant == "announced_dividend",
            "dividend_source": "local point-in-time files",
        }]).to_csv(target_dir / "run_manifest.csv", index=False)
    if announced_eligible:
        logger.info("announced 与 upper-bound 已生成；upper-bound 含前视，禁止正式结论。")
    else:
        logger.info("仅生成 upper-bound（含前视）；announced 因公告日不完整未生成。")


if __name__ == "__main__":
    main()
