# Google Colab 运行

先确保本目录的改动已经提交并推送到 GitHub，然后在 Colab 中运行：

```python
!git clone https://github.com/morewnutri/a-share-mainline-scanner.git
%cd /content/a-share-mainline-scanner
!ls -la colab
%run colab/run_scan.py
```

脚本默认扫描行业和概念源全集、不应用概念过滤规则，并把缓存与报告保存到 Google Drive。若遇到东方财富限流，把 `run_scan.py` 顶部的 `WORKERS` 改为 `1` 后重跑；成功缓存不会重复下载。

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

