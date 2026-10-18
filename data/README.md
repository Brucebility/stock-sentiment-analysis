# 数据目录

本目录放置原始数据，**不随仓库提交**。

| 文件 | 说明 | 来源 |
|---|---|---|
| `investor_sentiment_data .csv` | 股吧原始评论（含用户、时间、文本字段），文件名中间有一个空格 | 课程实践采集 |
| `public_stock_market_data_final.csv` | 个股日行情（日期、收盘价、收益率等） | 公开行情数据 |

## 运行产生的中转文件（自动生成，不提交）

`investor_sentiment_data_cleaned.csv` → `investor_sentiment_data_final.csv` → `investor_sentiment_data_processed.csv` → `investor_data_with_sentiment.csv` → `投资者情绪_最终特征表.csv` → `日度投资者情绪指标.csv`

## 说明

- 股吧数据来源于公开平台，此处仅作课程研究用途，不再分发原始抓取文件。
- 如需复现，请用你自己的同名数据集，字段结构需与脚本中的列名一致。
