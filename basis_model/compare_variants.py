"""比较 raw / announced / realized-upper-bound 三种基差口径。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from basis_model import config, strategy


def _index_dir(root: Path, variant_dir: str, code: str, name: str) -> Path:
    return root / variant_dir / "indices" / f"{code}_{name}"


def _load_equal_panels(root: Path, code: str, name: str) -> dict[str, pd.DataFrame]:
    folders = {
        "raw": "raw_basis",
        "announced": "dividend_adjusted",
        "upper_bound": "realized_dividend_upper_bound",
    }
    panels = {}
    for variant, folder in folders.items():
        path = _index_dir(root, folder, code, name) / "factor_panel.csv"
        if variant == "upper_bound" and not path.exists():
            continue
        if not path.exists():
            raise FileNotFoundError(f"缺少 {variant} 因子面板: {path}")
        panels[variant] = pd.read_csv(path, parse_dates=["date"], dtype={"index_code": str})
    keys = panels["raw"][["date", "contract"]].reset_index(drop=True)
    for variant, panel in panels.items():
        if len(panel) != len(keys) or not panel[["date", "contract"]].reset_index(drop=True).equals(keys):
            raise ValueError(f"{code} {variant} 与 raw 的日期/合约集合不一致，禁止比较")
    return panels


def _front_signal(panel: pd.DataFrame, code: str) -> pd.DataFrame:
    front = panel[panel["month_tag"] == "当月"].sort_values("date").drop_duplicates("date").copy()
    continuous = pd.Series("CONTINUOUS_FRONT", index=front.index)
    front["signal"] = strategy.map_by_index(
        code, front["f1_ann_basis"], continuous, front["date"],
        window=config.TIMING_WINDOW, pct=config.TIMING_PCT,
    )
    return front[["date", "signal"]]


def _monthly_return_delta(root: Path, code: str, name: str) -> pd.DataFrame:
    frames = []
    for label, folder in (("raw", "raw_basis"), ("announced", "dividend_adjusted"),
                          ("upper_bound", "realized_dividend_upper_bound")):
        path = _index_dir(root, folder, code, name) / "backtest_equity.csv"
        if not path.exists():
            continue
        eq = pd.read_csv(path, parse_dates=["date"])
        eq[f"return_{label}"] = eq["equity"].pct_change().fillna(0.0)
        frames.append(eq[["date", f"return_{label}"]])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner", validate="one_to_one")
    merged["month"] = merged["date"].dt.to_period("M").astype(str)
    monthly = merged.groupby("month")[[c for c in merged if c.startswith("return_")]].sum().reset_index()
    monthly.insert(0, "index_code", code)
    monthly.insert(1, "name", name)
    monthly["announced_minus_raw"] = monthly["return_announced"] - monthly["return_raw"]
    if "return_upper_bound" in monthly:
        monthly["upper_bound_minus_raw"] = monthly["return_upper_bound"] - monthly["return_raw"]
    return monthly


def compare_variants(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_root)
    raw_summary = pd.read_csv(root / "raw_basis" / "summary.csv", dtype={"index_code": str})
    announced_path = root / "dividend_adjusted" / "summary.csv"
    manifest_path = root / "dividend_adjusted" / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("缺少 announced 资格清单；请先运行 run_dividend_variants")
    manifest = pd.read_csv(manifest_path)
    eligible = manifest.get("formal_backtest_eligible", pd.Series([False])).fillna(False).astype(bool).all()
    if not eligible:
        reason = manifest.get("reason", pd.Series(["announced 不具备 point-in-time 资格"])).iloc[0]
        raise ValueError(f"禁止读取 announced 结果: {reason}")
    if not announced_path.exists():
        raise FileNotFoundError("缺少 announced 结果；本地分红三表未通过验收时禁止生成对比")
    rows, monthly_frames = [], []
    for item in raw_summary.itertuples(index=False):
        code, name = str(item.index_code).zfill(6), item.name
        panels = _load_equal_panels(root, code, name)
        raw, announced = panels["raw"], panels["announced"]
        season_raw = raw[pd.to_datetime(raw["date"]).dt.month.isin([5, 6, 7])]["adj_basis"]
        season_ann = announced[pd.to_datetime(announced["date"]).dt.month.isin([5, 6, 7])]["adj_basis"]
        raw_signal = _front_signal(raw, code)
        ann_signal = _front_signal(announced, code)
        signals = raw_signal.merge(ann_signal, on="date", suffixes=("_raw", "_announced"))
        active = (signals["signal_raw"] != 0) | (signals["signal_announced"] != 0)
        reversals = signals["signal_raw"].ne(signals["signal_announced"])
        row = {
            "index_code": code, "name": name,
            "n_panel_rows": len(raw),
            "season_mean_raw": season_raw.mean(), "season_mean_announced": season_ann.mean(),
            "season_std_raw": season_raw.std(), "season_std_announced": season_ann.std(),
            "season_variance_reduction_pct": 1 - season_ann.var() / season_raw.var() if season_raw.var() else np.nan,
            "signal_corr_raw_announced": signals["signal_raw"].corr(signals["signal_announced"]),
            "signal_reversal_days": int(reversals.sum()),
            "signal_reversal_share": float(reversals.mean()),
            "active_signal_reversal_share": float(reversals[active].mean()) if active.any() else 0.0,
            "mean_abs_div_points_announced": float(announced["div_points"].abs().mean()),
        }
        if "upper_bound" in panels:
            upper = panels["upper_bound"]
            row["mean_abs_div_points_upper_bound"] = float(upper["div_points"].abs().mean())
            row["upper_bound_contains_lookahead"] = True
        rows.append(row)
        monthly_frames.append(_monthly_return_delta(root, code, name))

    comparison = pd.DataFrame(rows)
    fifty_300 = comparison[comparison["index_code"].isin(["000016", "000300"])]["mean_abs_div_points_announced"].mean()
    ic_im = comparison[comparison["index_code"].isin(["000905", "000852"])]["mean_abs_div_points_announced"].mean()
    variance_condition = bool((comparison.loc[comparison["index_code"].isin(["000016", "000300"]), "season_variance_reduction_pct"] > 0).all())
    cross_index_condition = bool(fifty_300 > ic_im)
    supported = variance_condition and cross_index_condition
    comparison["claim_variance_condition_met"] = variance_condition
    comparison["claim_cross_index_condition_met"] = cross_index_condition
    comparison["dividend_pollution_claim_supported"] = supported
    comparison["conclusion"] = (
        "数据支持：分红季污染信号" if supported else "未观察到满足预设门槛的显著污染"
    )
    return comparison, pd.concat(monthly_frames, ignore_index=True)


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "output"
    comparison, monthly = compare_variants(root)
    comparison_dir = root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_dir / "basis_variants.csv", index=False)
    monthly.to_csv(comparison_dir / "basis_variants_monthly_returns.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
