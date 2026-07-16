# 本地 point-in-time 分红数据

支持 CSV、XLSX、XLS、Parquet。每类只能存在一个同名文件。

## 1. dividend_events

文件名：`dividend_events.*`

必需字段：

- `stock_code`：股票代码
- `index_code`：指数代码（000016/000300/000905/000852）
- `ann_date`：公告日；announced 防前视门控字段
- `ex_date`：除权除息日
- `cash_div`：每股现金分红

可选字段：`end_date`（报告期/分红年度，用于识别同一方案的公告更新）。

## 2. index_weights

文件名：`index_weights.*`

必需字段：

- `stock_code`
- `index_code`
- `trade_date`：权重快照生效日
- `weight_pct`：百分数权重，例如 2.5 表示 2.5%

## 3. stock_prices

文件名：`stock_prices.*`

必需字段：

- `stock_code`
- `trade_date`
- `close`：未复权收盘价

计算时权重与股价均只使用 `trade_date <= 估值日` 的最近快照；回测起点以前的
最近价格快照也会保留，以便起始阶段正确 as-of。

数据资格规则：

- 三张表或 `ex_date/cash_div/历史权重/as-of 价格`缺失：announced 和 upper-bound
  整体跳过。
- 基础字段齐全，但 `ann_date` 缺失或部分为空：announced 强制跳过；只生成明确标注
  含前视、禁止正式回测结论的 upper-bound。
- `ann_date` 完整有效：announced 和 upper-bound 均可生成。

程序不会用“除息日前 N 天”自动代理公告日。`run_manifest.csv` 会记录 announced
是否具备 point-in-time 正式回测资格，后续对比脚本会先检查该状态，避免误读旧结果。

运行：

```bash
python -m basis_model.run_dividend_variants
python -m basis_model.compare_variants
```
