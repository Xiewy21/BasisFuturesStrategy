from __future__ import annotations

import pandas as pd

from basis_model.data import LocalDividendProvider, get_dividend_adapter


def _write_local_dividend_files(tmp_path):
    pd.DataFrame({
        "stock_code": ["600000.SH"], "index_code": ["000300.SH"],
        "ann_date": ["2024-04-01"], "ex_date": ["2024-06-15"],
        "end_date": ["2023-12-31"], "cash_div": [1.0],
    }).to_csv(tmp_path / "dividend_events.csv", index=False)
    pd.DataFrame({
        "stock_code": ["600000.SH"], "index_code": ["000300.SH"],
        "trade_date": ["2024-03-29"], "weight_pct": [2.0],
    }).to_csv(tmp_path / "index_weights.csv", index=False)
    pd.DataFrame({
        "stock_code": ["600000.SH"], "trade_date": ["2024-04-01"], "close": [10.0],
    }).to_csv(tmp_path / "stock_prices.csv", index=False)


def test_local_dividend_provider_normalizes_three_tables(tmp_path):
    _write_local_dividend_files(tmp_path)
    provider = LocalDividendProvider(data_dir=tmp_path)
    ok, _ = provider.validate_access()
    assert ok
    dataset = provider.load_dataset("000300", "2024-01-01", "2024-12-31")
    assert set(dataset) == {"events", "weights", "prices"}
    assert dataset["events"].loc[0, "index_code"] == "000300"
    assert dataset["weights"].loc[0, "weight_pct"] == 2.0
    assert dataset["prices"].loc[0, "close"] == 10.0


def test_local_dividend_provider_safely_rejects_missing_files(tmp_path):
    ok, message = LocalDividendProvider(data_dir=tmp_path).validate_access()
    assert not ok
    assert "本地分红数据不可用" in message


def test_missing_announcement_still_allows_upper_bound_dataset(tmp_path):
    _write_local_dividend_files(tmp_path)
    events_path = tmp_path / "dividend_events.csv"
    events = pd.read_csv(events_path).drop(columns="ann_date")
    events.to_csv(events_path, index=False)

    provider = get_dividend_adapter("local", data_dir=tmp_path)
    ok, _ = provider.validate_access()
    assert ok
    dataset = provider.load_dataset("000300", "2024-01-01", "2024-12-31")
    assert dataset["events"]["ann_date"].isna().all()


def test_price_snapshot_before_backtest_start_is_retained(tmp_path):
    _write_local_dividend_files(tmp_path)
    pd.DataFrame({
        "stock_code": ["600000.SH"], "trade_date": ["2023-12-29"], "close": [9.5],
    }).to_csv(tmp_path / "stock_prices.csv", index=False)
    dataset = LocalDividendProvider(data_dir=tmp_path).load_dataset(
        "000300", "2024-01-01", "2024-12-31"
    )
    assert dataset["prices"].loc[0, "trade_date"] == pd.Timestamp("2023-12-29")
