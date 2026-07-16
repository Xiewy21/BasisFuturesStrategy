from __future__ import annotations

import io
import zipfile

import pandas as pd

from basis_model.data import CffexAkShareAdapter


def _monthly_zip() -> bytes:
    csv_text = (
        "合约代码,今开盘,最高价,最低价,成交量,成交金额,持仓量,持仓变化,今收盘,今结算,前结算,涨跌1,涨跌2,Delta\n"
        "IF2406,3500,3510,3490,10,1,20,0,3505,3504,3498,7,6,--\n"
        "IF2407,3510,3520,3500,10,1,20,0,3515,3514,3508,7,6,--\n"
        "IF2409,3520,3530,3510,10,1,20,0,3525,3524,3518,7,6,--\n"
        "IF2412,3530,3540,3520,10,1,20,0,3535,3534,3528,7,6,--\n"
        "小计,,,,40,4,80,0,,,,,,\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        # 2024-06-21 是 IF2406 的第三个周五；已到期合约不应占据当月标签。
        archive.writestr("20240621_1.csv", csv_text.encode("gb2312"))
    return buf.getvalue()


def test_cffex_month_parser_reads_official_schema():
    parsed = CffexAkShareAdapter._parse_month(_monthly_zip(), "IF")
    assert len(parsed) == 4
    assert set(parsed["contract"]) == {"IF2406", "IF2407", "IF2409", "IF2412"}
    assert parsed.loc[parsed["contract"] == "IF2407", "F"].item() == 3515


def test_expired_contract_is_removed_before_month_tagging(tmp_path, monkeypatch):
    adapter = CffexAkShareAdapter(cache_dir=tmp_path)
    monkeypatch.setattr(adapter, "_download_month", lambda *args, **kwargs: _monthly_zip())
    panel = adapter.load_futures("000300", "2024-06-21", "2024-06-21", refresh=True)
    assert "IF2406" not in set(panel["contract"])
    assert panel.loc[panel["contract"] == "IF2407", "month_tag"].item() == "当月"
    assert panel["date"].eq(pd.Timestamp("2024-06-21")).all()
