# 输出目录导航

日常查看按以下顺序即可：

```text
output/
├── raw_basis/                       # 当前正式 raw 口径
│   ├── summary.csv                  # 第一张表：主回测跨指数总览
│   ├── run_manifest.csv             # 数据源、区间、分红口径
│   ├── indices/                     # 各指数因子/回归/展仓等底层明细
│   ├── fixed_signals/               # 三个固定方向与基准
│   │   ├── summary.csv
│   │   └── <code>_<name>.csv
│   └── walkforward/                 # 动态选向纯样本外结果
│       ├── summary.csv
│       ├── selection_counts_summary.csv
│       ├── indices/<code>_<name>/   # metrics/equity/log/counts
│       └── cross_asset/             # IC训练→IM应用
├── dividend_adjusted/               # announced 口径；当前等待本地分红三表
├── comparisons/                     # raw/announced 跨口径汇总
└── archive/                         # 不属于当前正式流程的历史预览
```

`raw_basis/` 当前读取本地价格文件，不做分红调整，`div_points=0`。
`dividend_adjusted/` 只接受 point-in-time 合格的本地分红数据，不生成伪修正结果。

将三张本地分红长表放入 `input/local_dividend/`，再生成防前视 announced 和含前视上界：

```bash
python -m basis_model.run_dividend_variants
```

命令会先验证三张本地文件及字段；任一不可用时整体跳过，不生成
空表或伪修正结果。三套结果都存在后，运行：

```bash
python -m basis_model.compare_variants
```

对比结果会写入 `output/comparisons/`。
