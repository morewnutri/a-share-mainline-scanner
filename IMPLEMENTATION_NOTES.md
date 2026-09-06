# 估值链路升级说明

本版本基于 `main` commit `08873792e07f376f258144f660d23f1eb1fd05f1` 修改，目标是让估值结果在数据、模型或历史不足时主动降级或拒绝交易，而不是输出看似精确的价位。

## 已落实

- 修复 `valuation_config.json` 非法 JSON，修正 `600863` 为“内蒙华电”，补齐医疗器械、家电、公用事业模型族。
- `load_config()` 强制校验股票代码、名称、行业引用及支持的模型类型；未知行业返回 `MODEL_UNRESOLVED / NO_TRADE`。
- TTM 数值同时输出 method/confidence，区分 exact、Q1/H1/Q3 年化、报价源 PE 和缺失。
- 个股数据质量使用 TTM、增长、历史、现金流、行业、财报新鲜度六部分评分并映射 HIGH/MEDIUM/LOW/NO_VALUATION。
- 板块输出 aggregate/positive-profit PE、盈利与亏损市值占比，以及 PE/PB/PS 各自的覆盖率、来源与置信度；PE 盈利市值覆盖不足 70% 时禁用。
- ROE 优先使用 TTM 净利润/当前权益代理；年化披露 ROE 只作为低置信度回退。
- 增加现金转化、毛利率趋势、ROE 趋势；增长门槛按公用事业、周期、成长、成熟消费模型族区分。
- 历史倍数先按交易日排序和去重，再以自相关调整后的有效样本量进行 log-space empirical-Bayes shrinkage。
- AKShare/乐咕失败显式输出 history status/source/points/last date/confidence；AKShare 固定为 `1.18.38`，并增加字段合约测试。
- 对概念主题显式区分 boards（归类）与 valuation_boards（估值样本）。
- 公允价使用 PE/PB/PS/OCF 隐含价格的行业权重几何平均，输出模型分歧、公允价中心及区间。
- 自建 point-in-time 个股估值快照，保存当时的公允价与 `log(price/fair)` 残差；历史足够后使用 Q10/Q25/Q75/Q90 和 MAD Robust-Z。
- 买入价同时受 Q25 和绝对安全价值约束；安全边际随模型分歧、数据质量与周期风险变化，并限制在 10%~45%。
- 输出深度建仓、建仓、Q35 滞回退出、减仓、估值退出价，结合主线生命周期形成目标仓位、波动率缩放和最终动作。
- CLI 自动读取 `reports/latest/板块完整评分.csv`，并生成 `估值风险告警.csv` 及 Excel 告警工作表。

## 边界

- 免费数据源没有稳定、完整的资本开支与债务口径，因此当前现金流镜头明确叫 `OCF_YIELD_PROXY`，没有冒充精确 FCFF/DCF。
- rNPV、P/EV、银行剩余收益等模型需要额外的专业数据。未配置这些数据与模型前，系统会拒绝相应未知模型，而不会静默回退。
- point-in-time 残差需要持续运行积累；未达最低样本量时交易带标记为 `MODEL_PRIOR_BANDS`。

## 验证

```powershell
$env:PYTHONPATH='src'
python -m pytest -q -p no:cacheprovider
```

交付时共 36 项测试通过。
