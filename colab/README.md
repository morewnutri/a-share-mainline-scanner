# Google Colab 运行

先确保本目录的改动已经提交并推送到 GitHub，然后在 Colab 中运行：

```python
!git clone --branch codex/quant-v2-refactor https://github.com/morewnutri/a-share-mainline-scanner.git
%cd /content/a-share-mainline-scanner
!ls -la colab
%run colab/run_scan.py
```

脚本默认扫描行业和概念源全集，展示量化火种、确认主线、结构强势、切换、退潮、漏检诊断与横盘火种。政策、供需、订单等只显示为“未检查”的人工提醒。首次扫描即使没有历史快照，也能利用日线生成部分火种信号；切换仍需跨交易日快照。日线按“东方财富 → 同花顺 → 申万研究一级/二级行业”回退；真实主力资金不可用时会显示明确标注的 CMF 量价代理。脚本按 Noto CJK 字体文件绝对路径注册中文字体；若遇到限流，把 `run_scan.py` 顶部的 `WORKERS` 改为 `1` 后重跑。

可选上传 `/content/market_daily.csv` 提供独立全市场成交额和基准指数。`board_signals.csv` 不再参与评分。每次运行会把快照打包到 `/content/a-share-mainline-snapshots.zip`；请下载保存。下次上传后，把 `run_scan.py` 顶部的 `SNAPSHOT_ARCHIVE_IN` 设为上传文件路径。未导入时脚本会警告，`/content` 不保证跨会话保留。运行时会打印 Git 提交短哈希。

查看遗漏板块：

```python
import pandas as pd
path = "/content/a-share-mainline-results/遗漏板块明细.csv"
missing = pd.read_csv(path)
display(missing[["kind", "code", "name", "audit_status", "fetch_error"]])
```

查看完整性汇总：

```python
path = "/content/a-share-mainline-results/数据完整性审计.xlsx"
display(pd.read_excel(path, sheet_name="汇总"))
```

估值建议在 notebook 内运行（这样新增股票的 `input()` 可交互）：

```python
%run colab/run_valuation.py
```

估值输出位于 `/content/a-share-valuation-results/`。若要跨会话保留结果，请在会话结束前手动下载。
`colab/run_valuation.py` 会自动读取 `/content/a-share-mainline-results/板块完整评分.csv`，以免估值阶段一直显示 `UNKNOWN`。若直接运行命令行，请显式加上 `--mainline-csv /content/a-share-mainline-results/板块完整评分.csv`。板块估值若出现东方财富 502，属于上游接口失败，与四榜展示及主线阶段文件路径是两个独立问题。
