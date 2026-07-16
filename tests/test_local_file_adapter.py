from __future__ import annotations

import pandas as pd
import pytest

from basis_model.data import LocalFileAdapter, get_adapter, merge_panel


def _write_inputs(tmp_path):
    futures = pd.DataFrame(
        {
            "交易日期": ["20240620", "20240620", "20240621", "20240621"],
            "合约代码": ["IF2406.CFX", "IF2407.CFX", "IF2406.CFX", "IF2407.CFX"],
            "收盘价": [3500.0, 3510.0, 3505.0, 3515.0],
        }
    )
    indices = pd.DataFrame(
        {
            "date": ["2024-06-20", "2024-06-21"],
            "index_code": ["000300.SH", "000300.SH"],
            "close": [3490.0, 3500.0],
        }
    )
    futures.to_csv(tmp_path / "futures_all_contracts.csv", index=False)
    indices.to_csv(tmp_path / "index_daily.csv", index=False)


def test_local_adapter_normalizes_and_merges_csv(tmp_path):
    _write_inputs(tmp_path)
    adapter = LocalFileAdapter(data_dir=tmp_path)
    futures = adapter.load_futures("000300", "2024-06-20", "2024-06-21")
    index = adapter.load_index("000300", "2024-06-20", "2024-06-21")
    panel = merge_panel(futures, index)

    assert set(panel["contract"]) == {"IF2406", "IF2407"}
    # 6月21日 IF2406 已到期，IF2407 必须重新标为当月。
    expiry_day = panel[panel["date"] == pd.Timestamp("2024-06-21")]
    assert set(expiry_day["contract"]) == {"IF2407"}
    assert expiry_day["month_tag"].item() == "当月"
    assert expiry_day["S"].item() == 3500.0


def test_local_adapter_rejects_conflicting_duplicates(tmp_path):
    _write_inputs(tmp_path)
    path = tmp_path / "futures_all_contracts.csv"
    futures = pd.read_csv(path)
    futures.loc[len(futures)] = ["20240620", "IF2407.CFX", 9999.0]
    futures.to_csv(path, index=False)

    adapter = LocalFileAdapter(data_dir=tmp_path)
    with pytest.raises(ValueError, match="冲突值"):
        adapter.load_futures("000300", "2024-06-20", "2024-06-21")


def test_local_adapter_is_registered():
    assert isinstance(get_adapter("local"), LocalFileAdapter)


def test_local_adapter_reads_wind_wide_continuous_workbook(tmp_path):
    raw = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-06-20"), pd.Timestamp("2024-06-21")],
            "000300.SH": [3490.0, 3500.0],
            "IF00.CFE": [3500.0, 3505.0],
            "IF01.CFE": [3510.0, 3515.0],
            "IF02.CFE": [3520.0, 3525.0],
            "IF03.CFE": [3530.0, 3535.0],
        }
    )
    path = tmp_path / "价格.xlsx"
    with pd.ExcelWriter(path) as writer:
        raw.to_excel(writer, index=False, startrow=4)

    adapter = LocalFileAdapter(data_dir=tmp_path)
    futures = adapter.load_futures("000300", "2024-06-20", "2024-06-21")
    index = adapter.load_index("000300", "2024-06-20", "2024-06-21")

    assert futures["date"].nunique() == 2
    assert futures.groupby("date").size().eq(4).all()
    assert futures.groupby("date")["month_tag"].nunique().eq(4).all()
    assert set(index["S"]) == {3490.0, 3500.0}
