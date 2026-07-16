"""本地期货/指数文件的回测前验收。

用法：
    python -m basis_model.validate_local_data
    python -m basis_model.validate_local_data --data-dir /absolute/path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from basis_model import config as config_mod
from basis_model import data as data_mod


def validate(data_dir: str | Path) -> pd.DataFrame:
    adapter = data_mod.LocalFileAdapter(data_dir=data_dir)
    rows = []
    for index_cfg in config_mod.INDEX_CONFIGS:
        futures = adapter.load_futures(
            index_cfg.index_code, index_cfg.data_start, config_mod.END_DATE
        )
        index = adapter.load_index(
            index_cfg.index_code, index_cfg.data_start, config_mod.END_DATE
        )
        panel = data_mod.merge_panel(futures, index)
        futures_days = futures["date"].nunique() if not futures.empty else 0
        index_days = index["date"].nunique() if not index.empty else 0
        matched_days = panel["date"].nunique() if not panel.empty else 0
        rows.append(
            {
                "index_code": index_cfg.index_code,
                "product": index_cfg.product,
                "name": index_cfg.name,
                "requested_start": index_cfg.data_start,
                "actual_start": panel["date"].min() if not panel.empty else pd.NaT,
                "actual_end": panel["date"].max() if not panel.empty else pd.NaT,
                "futures_rows": len(futures),
                "contracts": futures["contract"].nunique() if not futures.empty else 0,
                "futures_days": futures_days,
                "index_days": index_days,
                "matched_days": matched_days,
                "date_match_ratio": matched_days / futures_days if futures_days else 0.0,
                "front_month_coverage": (
                    futures.loc[futures["month_tag"] == "当月", "date"].nunique()
                    / futures_days
                    if futures_days
                    else 0.0
                ),
                "status": "PASS"
                if matched_days > 0
                and matched_days == futures_days
                and futures.loc[futures["month_tag"] == "当月", "date"].nunique()
                == futures_days
                else "CHECK",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="验收本地股指期货/现货指数数据")
    parser.add_argument(
        "--data-dir",
        default=str(data_mod.LOCAL_DATA_DIR),
        help="本地数据目录，默认 input/local_data",
    )
    args = parser.parse_args()
    report = validate(args.data_dir)
    target = Path(args.data_dir) / "local_data_validation.csv"
    report.to_csv(target, index=False)
    print(report.to_string(index=False))
    print(f"\n验收报告: {target}")
    if not report["status"].eq("PASS").all():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
