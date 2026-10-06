# 主线研究证据接口

`--research-signals` 接受宽表 CSV。每行是一个板块在某一时点新出现的一组证据；同一板块可有多行，后续稀疏行不会清空此前已知的其他字段。

必填：`kind`（`industry`/`concept`）、`code`（与扫描器代码一致）、`available_at`（模型真正可得时间）。建议同时填 `published_at`、`source_url`、`evidence_note`。无时区的时间按北京时间解释；带时区的时间转换为北京时间。`available_at` 不得早于 `published_at`；扫描时只采纳 `available_at <= 扫描时刻` 且不晚于行情日期的记录。历史回测应保留每天实际获得的文件版本，不可拿今天回填的历史共识或当前概念成分冒充过去数据。

所有可选信号均在 **0–1** 范围，`0.5` 表示中性；缺失请留空，不能填 `0.5` 假装有数据。高值代表该方向更有利，唯退潮和海外风险字段高值代表风险较高。数值可来自人工有证据的研究记录或合法数据源的滚动分位映射；来源和原始值宜单独保留。评分只对可用维度重新加权，同时输出覆盖率。报告中 30/25/20/15/10 和 25/20/15/15/10/10/5 权重均为待滚动验证的先验。

| 组别 | CSV 可选列 | 用途 |
| --- | --- | --- |
| 盈利 | `eps_revision_fy1`, `eps_revision_fy2`, `revision_breadth`, `revenue_revision`, `profit_growth`, `gross_margin_trend` | FY1/FY2 预测修订、上修广度、营收、利润与毛利率变化 |
| 产业 | `industry_demand`, `industry_price`, `industry_orders`, `industry_capacity`, `industry_inventory` | 销量、价格/价差、订单、产能和库存等行业特定高频指标 |
| 政策 | `policy_level`, `policy_specificity`, `policy_implementation`, `policy_economic_size` | 政策层级、具体执行、财政/税收/项目规模；四项齐全时使用几何平均，避免只凭口号得高分 |
| 催化 | `catalyst_chain`, `catalyst_frequency`, `catalyst_forward` | 政策→产品→订单→产能→业绩的连续链条、频率和后续日程 |
| 预期差 | `earnings_surprise`, `guidance_surprise` | 实际业绩/预告相对于**当时预期**的差值，不能只看绝对增速 |
| 龙头梯队 | `leader_strength`, `leader_turnover`, `leader_capacity`, `leader_sector_link`, `leader_lead_lag`, `leader_recovery`, `down_market_resilience`, `rebound_leadership`, `middle_army`, `follower_tier`, `upstream_diffusion`, `institutional_participation` | 龙头相对强度、成交额、市值容量、与板块相关性和领先关系、回撤修复、中军/补涨/上下游扩散、大市值/机构参与、逆势抗跌与修复领先 |
| 成分股广度 | `breadth_ma20`, `breadth_high20`, `median_constituent_return` | MA20 之上、新高比例、中位数收益；扫描器原有 `breadth` 为当日上涨比例 |
| 关注度/情绪 | `attention_rank`, `news_attention`, `search_attention`, `dragon_tiger_seat`, `limit_up_breadth`, `popularity_rank`, `relay_success_rate` | 热榜、新闻/搜索、龙虎榜公开席位、涨停扩散和接力成功率；关注度只占确认层低权重，不当成预期收益。龙虎榜仅为异常交易局部样本 |
| 指数共振 | `index_resonance` | 大盘调整时抗跌、修复时领先；已有 5/10/20 日相对指数收益也会进入确认层 |
| 分时与盘口 | `volume_price_alignment`, `aggressive_buying`, `tape_acceptance`, `orderbook_liquidity` | 量价配合、主动买盘、承接、盘口流动性；须有合规分钟/Tick/Level-2 数据才填写，日线不能推断逐笔主动成交 |
| 退潮 | `leader_divergence`, `breakout_failure_rate`, `relay_failure_rate`, `edge_catchup`, `rotation_speed`, `high_volume_stall`, `catalyst_exhaustion` | 龙头背离、突破/接力失败、边缘补涨、轮动加速、放量滞涨、重大催化兑现 |
| 海外风险 | `export_restriction`, `tariff_exposure`, `entity_risk`, `customer_exposure`, `supplier_exposure`, `substitution_benefit` | 负面暴露与国产替代受益分别录入；`geo_net_exposure` 单独展示，不因“制裁”关键词自动判断方向 |

`revision_breadth` 可先计算 `(EPS上调公司数 − EPS下调公司数) / 有预测覆盖公司数`，再用截至当时的历史分位映射到 0–1。覆盖公司数也应另存以审计。多个行业的高频指标含义不同，先按行业各自口径标准化。成分股指标只可使用当时的成分名单；没有历史名单的概念板块不要回填。证券停牌应标记为合法状态，不当成抓取失败。

`--market-history` CSV：`date,market_amount,benchmark_close`。`benchmark_close` 可省略；缺少时相对强度使用同类板块截面代理并在原字段中保留口径。`market_amount` 必须是独立汇总的同单位全 A 成交额；每个概念内部先对股票去重，不同概念之间允许重叠，所有概念占比之和因此可以超过 100%。可选字段 `stock_amount_sum` 参与对账，默认超过 5% 误差停止评分。`expected_stock_count`, `valid_quote_count`, `suspended_count` 计算价格覆盖率（合法停牌计入覆盖）；`theme_mapped_count` 计算主题覆盖率；`financial_report_due_count`, `financial_report_available_count` 计算财报覆盖率；`consensus_covered_count` 计算一致预期覆盖率。这些统计会输出至 `全A数据质量.csv`。

四榜均保留全部板块。`potential_score` 缺失表示没有基本面证据，并不等于分数低。`market_confirmation_score`、`exhaustion_score` 也需结合相应 `*_coverage` 看；缺失的 Level-2、历史一致预期与官方海外事件映射不会从股价猜测。
