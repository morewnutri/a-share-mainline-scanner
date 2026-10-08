# A股量化主线扫描器（quant-v2）

扫描行业和概念板块的**已确认主线、量化火种、横盘潜伏、结构强势、主线切换与退潮风险**。分数只使用可自动观测、可核对交易日、可复算的行情数据。政策、供需、订单、业绩预期和产业事件以“未检查”的固定清单展示，不参与评分、排名或生命周期。

## 安装与运行

```powershell
python -m pip install -e .
mainline-scanner
```

常用参数：

```powershell
mainline-scanner --board-types industry concept --workers 3
mainline-scanner --market-history data/input/market_daily.csv
mainline-scanner --snapshot-dir data/snapshots --backtest
mainline-scanner --historical-replay  # 逐交易日重建基础量化信号
mainline-scanner --structure-candidates 0  # 关闭在线成分股验证
```

默认扫描所有板块。历史日线依次尝试东方财富、同花顺、申万及可选的 BaoStock 合成回退。合成指数会标记来源，不可仅凭其价格信号升级为 `Mainline`。申万请求要求正常 TLS 证书验证。

可选的 `--market-history` CSV 至少包含 `date,market_amount`；可加 `benchmark_close`。成交额应与板块日线同单位。如果提供 `stock_amount_sum`，默认要求它与全市场成交额误差不超过 5%。没有这个文件时，真实全市场成交占比保持缺失，量能以板块自身历史变化衡量。

`--research-signals` 仅为旧命令兼容入口，会提示弃用，其文件内容完全不参与计算。`board_signals.csv` 不再是生成火种榜的前提。

## 分数与确认条件

- 主线分按趋势、相对强度、量能、广度、持续性分组，再合并组分。每个分数都有原始分、有效权重覆盖率和排序分：`rank_score = 50 + (raw_score - 50) × coverage`。缺失因子不会填成中性观测。
- 火种分使用历史日线重建的排名变化、强势持续性、量能与加速度；有跨交易日快照时补充历史确认分和份额变化。首次运行仍可生成火种候选。
- `Mainline` 还要求 5 日趋势上行、10 日绝对收益为正、已知广度不低于 45%、主线指标覆盖率至少 65%、主线分至少 80，且不是合成指数。`mainline_blockers` 列逐板块列出未达标项。
- 候选板块的个股结构在收盘后尝试自动验证。全市场个股快照只取一次，成分股只拉取候选；映射与有效行情覆盖率均需达到 70%。来源不可用、日期不一致或覆盖不足时，结构分保持缺失。今天的成分股名单不会用于历史回放。
- `name`、`lifecycle`、`mainline_score` 保持与估值模块兼容；若评分覆盖不足，估值接口不会把 `Mainline` 当成可信阶段。

这些阈值是待滚动回测校准的初始规则，不是收益预测。`主线判断报告.md`、Excel 和 `板块完整评分.csv` 会显示覆盖率、未入主线原因、结构验证状态及人工核查提醒。无合格主线时，报告明确写“暂无满足确认条件的主线”，不会用普通高分板块回填。

## 日期、缓存与快照

扫描以交易所已完成的收盘交易日为准。盘中扫描仍使用前一已完成交易日，不混入当日未收盘数据。历史缓存除文件时效外，还核对请求起点、预期截止交易日和实际最后行情日期。休市期间多次运行不会制造新的交易日变化。

快照保存 `market_as_of`、`captured_at`、`run_mode`、`model_version`、`config_hash`、`universe_hash` 和数据指纹。跨日比较只使用前一实际交易日、相同模型和配置的收盘快照。同日重复运行会保留文件，但不会算作新的市场观察日。回测只使用完整未来窗口；不足的标为 `censored`，成功标签要求未来正式 `lifecycle == Mainline`。

Colab 脚本位于 `colab/run_scan.py`。它会输出快照 ZIP；请下载并在下次运行前设置 `SNAPSHOT_ARCHIVE_IN`。Colab 的 `/content` 会话结束后通常不会保留历史快照。

## 验证

```powershell
python -m pytest -q
```

测试覆盖缺失评分、交易日快照、缓存区间、正式生命周期标签、未来窗口删失、主观字段不影响评分、结构覆盖门槛及估值兼容。未附完整历史行情与当时的成分股映射，因此项目尚未宣称量化火种在电池事件或全历史样本中有更好的命中率；需用逐交易日历史回放评估误报与漏检。

`--historical-replay` 只使用截至每个交易日的板块日线和已提供的市场日线，排除当前资金流和当前成分股数据。它使用**当前板块目录**，因此仍有板块存续偏差；完整研究级回测需要历史目录版本。
