# 本地行情数据切口

将已获取的数据放到本目录。支持 `.csv`、`.xlsx`、`.xls`、
`.parquet`。日期可以是 `YYYYMMDD`、`YYYY-MM-DD` 或 Excel 日期。

## 方式一：两个合并文件（推荐）

1. `futures_all_contracts.xlsx`（或同名 CSV/Parquet）
   - 必需字段：交易日期、合约代码、收盘价
   - 兼容英文：`trade_date/date`、`contract/symbol/ts_code`、`close`
   - 合约代码：`IH2409`、`IF2409`、`IC2409`、`IM2409`，
     也允许 `.CFX` / `.CFFEX` 后缀。

2. `index_daily.xlsx`（或同名 CSV/Parquet）
   - 必需字段：交易日期、指数代码、收盘价
   - 兼容英文：`trade_date/date`、`index_code/ts_code/symbol`、`close`
   - 指数代码：`000016`、`000300`、`000905`、`000852`，
     也允许 `000300.SH` 这类后缀。

## 方式二：四个指数文件

期货仍使用 `futures_all_contracts.*`；指数可拆成：

- `index_000016.xlsx`
- `index_000300.xlsx`
- `index_000905.xlsx`
- `index_000852.xlsx`

拆分文件只需日期和收盘价，指数代码从文件名推断。

## 方式三：单个 Excel 工作簿

文件名为 `market_data.xlsx`，包含两个 sheet：

- `futures`
- `indices`

## 方式四：Wind 宽表连续合约

也支持当前目录中的 `价格.xlsx`：第 5 行为机器字段名，现货列使用
`000016.SH` 等代码，期货列使用 `IH00.CFE` 至 `IH03.CFE` 等连续合约代码。
读取时会按交易日将 00/01/02/03 映射为当时在市的当月、次月、当季、下季合约。

## 运行

先验收数据：

```bash
python -m basis_model.validate_local_data
```

四个指数均为 `PASS` 后，将 `basis_model/config.py` 里的主数据源改为：

```python
DATA_SOURCE_PRIMARY = "local"
```

然后执行：

```bash
python -m basis_model.run
```

也可以不改配置，在 Python 中显式指定任意路径：

```python
from basis_model.data import LocalFileAdapter

adapter = LocalFileAdapter(
    futures_path="/absolute/path/to/futures.xlsx",
    index_path="/absolute/path/to/indices.xlsx",
    futures_sheet="futures",  # 单 sheet 文件可省略
    index_sheet="indices",
)
```

## 数据检查

- 同一交易日+合约代码出现不同收盘价时会直接报错。
- 同一交易日+指数代码出现不同收盘价时会直接报错。
- 到期合约在到期日后会被排除，期限标签重新按在市合约计算。
- 这两类数据只支持原始基差；分红修正仍需历史权重和分红公告。
