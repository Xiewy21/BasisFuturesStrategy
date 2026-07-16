"""basis_model —— 股指期货基差研究/回测框架。

【定位声明】
本包是研究/回测工具，用于辅助分析股指期货基差的结构、统计规律与情景预测，
**不是**可直接对接交易系统的实盘执行工具。所有"信号""预测""回测收益"均基于
历史数据和简化假设，实盘使用前需要独立验证、加入真实成本/滑点/流动性约束。

【四层流水线】
    data(数据层) -> dividend(分红/结构层) -> factors+regression(统计层)
        -> strategy(策略/回测层) -> forecast(预测层)

各模块顶部 docstring 标注方法论性质：结构建模 / 统计建模 / 规则回测 / 定性判断。

【三条纪律（贯穿全部模块）】
    1. 顺序刚性：剔分红 -> 造因子 -> 回归 -> 策略映射 -> 预测，不可跳步、不可乱序。
    2. 防前视：future_return / y(收敛率) 只允许进入训练环节，预测路径严禁读取，
       用断言 (assert_no_lookahead) 与单测强制检查。
    3. 硬软分离：forecast 层返回结果必须显式区分 from_regression（硬，统计外推）
       与 from_qualitative（软，定性判断），不得混为一谈。
"""

__version__ = "0.1.0"

__all__ = [
    "data",
    "dividend",
    "factors",
    "regression",
    "strategy",
    "forecast",
    "report",
]
