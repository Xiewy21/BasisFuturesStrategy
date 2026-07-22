# 输出目录导航

日常查看按以下顺序即可：

```text
output/
├── factor_research/                 # 纯因子测算；因子与未来标签分离
├── raw_basis/                       # 当前正式 raw 口径
│   ├── summary.csv                  # 第一张表：主回测跨指数总览
│   ├── run_manifest.csv             # 数据源、区间、分红口径
│   ├── indices/                     # 各指数因子/回归/展仓等底层明细
│   ├── fixed_signals/               # 三个固定方向与基准
│   │   ├── summary.csv
│   │   └── <code>_<name>.csv
│   ├── walkforward/                  # 动态选向纯样本外结果
│   │   ├── summary.csv
│   │   ├── selection_counts_summary.csv
│   │   ├── indices/<code>_<name>/   # metrics/equity/log/counts
│   │   └── cross_asset/             # IC训练→IM应用
│   └── review/                      # 基差复盘四模块（见下方说明）
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

## 基差复盘（output/<variant>/review/）

四个模块只读已有的 `factor_panel.csv`，互不阻塞，可以按需单独运行：

```bash
python -m basis_model.term_structure       # 期限结构 PCA：term_vector / pca_loadings /
                                            # pca_explained_variance / regression_pca_vs_f3
python -m basis_model.basis_center         # 中枢与异常：basis_center / anomaly_events / 十年中枢图
python -m basis_model.convergence_prob     # 收敛概率：convergence_prob / 热力图
python -m basis_model.neutral_hedge_timing # 中性策略对冲择时：hedge_timing / cost_saving_summary
```

参数集中在 `config.ReviewConfig` / `config.DEFAULT_REVIEW_CONFIG`，改口径不用动模块代码。
