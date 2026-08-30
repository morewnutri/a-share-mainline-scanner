# Google Colab 运行

先确保本目录的改动已经提交并推送到 GitHub，然后在 Colab 中运行：

```python
!git clone https://github.com/morewnutri/a-share-mainline-scanner.git
%cd /content/a-share-mainline-scanner
!ls -la colab
%run colab/run_scan.py
```

脚本默认扫描行业和概念源全集、不应用概念过滤规则。缓存保存在 Google Drive，表格和两张图片直接显示在 Colab 输出区域，不创建下载任务。日线按“东方财富 → 同花顺 → 申万研究一级/二级行业”回退；真实主力资金不可用时会显示明确标注的 CMF 量价代理。脚本按 Noto CJK 字体文件绝对路径注册中文字体；若仍遇到限流，把 `run_scan.py` 顶部的 `WORKERS` 改为 `1` 后重跑。

查看遗漏板块：

```python
import pandas as pd
path = "/content/drive/MyDrive/a-share-mainline-scanner/reports/latest/遗漏板块明细.csv"
missing = pd.read_csv(path)
display(missing[["kind", "code", "name", "audit_status", "fetch_error"]])
```

查看完整性汇总：

```python
path = "/content/drive/MyDrive/a-share-mainline-scanner/reports/latest/数据完整性审计.xlsx"
display(pd.read_excel(path, sheet_name="汇总"))
```
