import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ========================= 1. 加载带情绪得分的数据 =========================
# 这是 4.3.3 生成的文件
df = pd.read_csv("data/investor_data_with_sentiment.csv", encoding="utf-8-sig")
print("加载数据完成，共 {} 条评论".format(len(df)))

# 确保日期格式正确
df["date"] = pd.to_datetime(df["date"]).dt.date

# ========================= 2. 日度情绪指标聚合 =========================
daily_sentiment = df.groupby("date")["sentiment_score"].agg(
    investor_mean="mean",        # 日情绪均值
    investor_median="median",    # 日情绪中位数
    investor_std="std",          # 情绪分歧（标准差）
    investor_count="count",      # 评论数（关注度）
    # 正向/负向占比
    investor_pos_ratio=lambda x: (x > 0).mean(),
    investor_neg_ratio=lambda x: (x < 0).mean(),
    # 极端情绪（狂热/恐慌）
    investor_extreme=lambda x: (abs(x) > 0.7).mean()
).reset_index()

print("✅ 日度情绪指标构建完成")

# ========================= 3. 构建滞后特征（LAG 1/3/5） =========================
daily_sentiment = daily_sentiment.sort_values("date").reset_index(drop=True)

# 滞后1日、3日、5日情绪（和媒体层保持一致）
daily_sentiment["investor_lag1"] = daily_sentiment["investor_mean"].shift(1)
daily_sentiment["investor_lag3"] = daily_sentiment["investor_mean"].shift(3)
daily_sentiment["investor_lag5"] = daily_sentiment["investor_mean"].shift(5)

# 去掉缺失值（滞后导致的前几行）
daily_sentiment = daily_sentiment.dropna().reset_index(drop=True)
print("✅ 滞后特征（lag1/3/5）构建完成")

# ========================= 4. 特征标准化（用于建模） =========================
# 需要标准化的列（非比例、非计数特征）
scale_cols = [
    "investor_mean", "investor_median", "investor_std",
    "investor_extreme", "investor_lag1", "investor_lag3", "investor_lag5"
]

scaler = StandardScaler()
daily_sentiment[scale_cols] = scaler.fit_transform(daily_sentiment[scale_cols])
print("✅ 情绪特征标准化完成")

# ========================= 5. 保存最终特征文件 =========================
daily_sentiment.to_csv("data/投资者情绪_最终特征表.csv", index=False, encoding="utf-8-sig")
print("✅ 最终文件已保存：投资者情绪_最终特征表.csv")

# ========================= 6. 输出预览 =========================
print("\n" + "="*60)
print("        投资者情绪特征工程 构建完成")
print("="*60)
print("最终特征列表：")
for col in daily_sentiment.columns:
    print("-", col)
print("="*60)
print("\n🎉 4.3.4 全部完成！可直接用于第5章建模！")