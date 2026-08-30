# A股主线生命周期扫描器

项目同时保留两条识别路径：

1. **主线确认层**：根据板块趋势、相对强弱、资金、广度、量能与拥挤度识别已经扩散的主线；
2. **火种发现层**：保存每次评分快照，优先观察排名跃迁、广度扩张、板块成交额份额迁移和同口径资金强度变化，尽量在过去 5 日涨幅仍不高时发现 `Seed / Ignition`。

生命周期输出为：

```text
Dormant → Seed → Ignition → Diffusion → Mainline → Crowded / Decay
```

`candidate_score` 为兼容 1.x 保留，等同于新的 `confirmation_score`。真正用于早期雷达的是 `ignition_score`。

## 安装与运行

```powershell
python -m pip install -e .
mainline-scanner
```

也可以直接运行模块：

```powershell
python -m mainline_scanner.cli --board-types industry concept --workers 3
```

常用参数：

```powershell
# 只扫描行业
mainline-scanner --board-types industry

# 忽略缓存并重抓
mainline-scanner --refresh

# 每类只扫描前 10 个，用于调试
mainline-scanner --limit 10

# 启用 BaoStock 行业合成回退
mainline-scanner --baostock-mode industry

# 行业和概念均尝试按成分股合成；首次运行明显更慢
mainline-scanner --baostock-mode all --baostock-max-constituents 24

# 扫描后同时回放已有历史快照
mainline-scanner --backtest
```

## 实时与历史缓存

实时板块列表和资金流默认缓存 **5 分钟**，历史日线默认缓存 **24 小时**，两者不再共用原来的 8 小时 TTL。东方财富实时域名 `push2` 优先，`push2delay` 只作兜底。

```powershell
mainline-scanner --snapshot-cache-minutes 3 --cache-hours 24
```

每次运行默认在 `data/snapshots/` 保存带时间戳的压缩 CSV。同一日多次运行可形成盘中 `breadth_delta_intraday` 和 `amount_share_delta_intraday`；跨日运行可形成：

- `confirmation_score_rank_velocity_1d/3d`
- `confirmation_score_delta_1d`
- `breadth_delta_1d`
- `amount_share_delta_1d`
- `snapshot_history_coverage`

首次运行没有历史轨迹，`ignition_score` 只能使用当日异常特征；连续保存快照后才具备完整的早期识别信息。

## 数据源与缺失回退

板块日线依次尝试：

| 顺序 | 数据源 | 适用范围 | 说明 |
| --- | --- | --- | --- |
| 1 | 东方财富 | 行业、概念 | 原始板块指数日线；连续探测 3 个板块后才判定端点整体不可用 |
| 2 | 同花顺 | 行业、概念 | 按标准化名称映射备用指数 |
| 3 | 申万研究 | 一级/二级行业 | 申万官方行业指数 |
| 4 | BaoStock 成分股等权合成 | 行业；可选概念 | 不是原始板块指数，明确标记合成来源和有效成分股覆盖率 |

BaoStock 不提供东方财富概念指数，不能直接补齐所有 `BKxxxx`。本项目使用它的个股日线和行业分类构造等权合成指数；`--baostock-mode all` 还会先获取概念成分股，再用 BaoStock 个股日线合成。为了控制免费接口压力，每个板块默认固定抽取最多 24 只成分股，并共享个股缓存。

因此合成回退适合“让板块不完全缺席”和交叉确认，不应与原始指数点位混为一谈。完整性审计新增：

- `history_source=BaoStock成分股等权合成`
- `synthetic_constituents`
- `synthetic_coverage`

`--baostock-mode` 默认关闭，避免全量扫描首次运行因数千只个股请求而耗时过长。建议先用 `industry`，确有需要再使用 `all`。

## 资金口径修正

真实主力净流入占比和 CMF 量价代理不再混在同一个横截面直接排名：

- 每种来源分别标准化；
- CMF 代理按 0.55 置信度向中性值收缩；
- 真实资金变化采用“今日净流入占比 - 5 日每日均值”；
- CMF 变化只与 CMF 比较；不同来源组合标记为不可比。

输出保留 `flow_*_source`、`flow_*_confidence` 和 `flow_acceleration_source`，便于二次筛选。

## 火种分与确认分

`ignition_score` 主要使用：

- 1/3 日火种排名跃迁；
- 确认分变化；
- 跨日和盘中广度增量；
- 板块成交额同类份额增量；
- 同口径资金强度变化；
- 价格加速度、当日异常和量能扩张。

过去 5/10 日已经大涨会对火种分扣分。`mainline_score` 继续承担主线确认，`confirmation_score` 保留原有短趋势确认逻辑。

注意：当前版本已落地板块轨迹型火种层，但“全 A 个股异常 → 概念反向投票 → 重叠概念聚类”仍属于下一阶段，不能把当前火种分理解成完整的个股异常聚类引擎。

## 历史回放

积累至少数日快照后运行：

```powershell
mainline-backtest --snapshot-dir data/snapshots --output-dir reports/backtest
```

输出 `火种信号回放明细.csv` 和 `火种信号回放汇总.csv`，评估：

- `precision_at_10`
- `false_start_rate`
- `median_lead_time_sessions`
- `median_alert_ret_5d`
- `mean_forward_rs_3d/5d/10d`

这是对扫描器“是否提前发现”的评估，不是买卖收益回测。

## 输出文件

默认输出到 `reports/latest/`：

- `板块完整评分.csv`
- `板块主线扫描.xlsx`（含“火种雷达”工作表）
- `主线雷达.png`
- `领先板块走势.png`
- `主线判断报告.md/.html`
- `数据完整性审计.xlsx`
- `遗漏板块明细.csv`

## Google Colab

```python
!git clone https://github.com/morewnutri/a-share-mainline-scanner.git
%cd /content/a-share-mainline-scanner
%run colab/run_scan.py
```

`colab/run_scan.py` 会把评分快照与行情缓存一起持久化到 Google Drive。可在脚本顶部设置 `BAOSTOCK_MODE = "industry"` 或 `"all"`。

## 使用边界

免费网页接口可能变更或限流；程序有重试、缓存、跨源回退和逐板块审计。概念板块高度重叠，成交额份额适合观察同一板块随时间的变化，不代表互斥市场份额。结果是发现与排序工具，不构成投资建议。

接口说明可参考 [AKShare 股票数据文档](https://akshare.akfamily.xyz/data/stock/stock.html) 与 [BaoStock 官方站点](http://baostock.com/)。
