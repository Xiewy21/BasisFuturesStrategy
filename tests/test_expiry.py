"""cffex_expiry / third_friday 到期日计算的单测。"""

from datetime import date

import pandas as pd
import pytest

from basis_model.data import cffex_expiry, days_to_expiry, parse_contract, third_friday


def test_third_friday_september_2024():
    # 2024-09-01 是周日，第一个周五是 09-06，第三个周五是 09-20
    assert third_friday(2024, 9) == date(2024, 9, 20)


def test_third_friday_march_2025():
    # 2025-03-01 是周六，第一个周五是 03-07，第三个周五是 03-21
    assert third_friday(2025, 3) == date(2025, 3, 21)


def test_cffex_expiry_if_contract():
    assert cffex_expiry("IF2409") == date(2024, 9, 20)


def test_cffex_expiry_lowercase_and_suffix():
    assert cffex_expiry("if2409") == date(2024, 9, 20)
    assert cffex_expiry("IF2409.CFX") == date(2024, 9, 20)


def test_cffex_expiry_is_always_friday():
    for contract in ["IH2501", "IF2506", "IC2512", "IM2503"]:
        d = cffex_expiry(contract)
        assert d.weekday() == 4  # Friday


def test_cffex_expiry_rolls_to_next_trade_day_when_friday_is_holiday():
    # 2018-02-16 是第三个周五且春节休市，下一交易日为 2018-02-22。
    calendar = pd.to_datetime(["2018-02-14", "2018-02-22", "2018-02-23"])
    assert cffex_expiry("IF1802", trade_dates=calendar) == date(2018, 2, 22)


def test_parse_contract_extracts_product_year_month():
    product, year, month = parse_contract("IC2411")
    assert product == "IC"
    assert year == 2024
    assert month == 11


def test_parse_contract_rejects_garbage():
    with pytest.raises(ValueError):
        parse_contract("not-a-contract")
