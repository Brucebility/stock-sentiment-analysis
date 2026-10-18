import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from gensim import corpora, models

# ---------------------- 1. 只读 2000 行，绝对不炸内存 ----------------------
data_path = "data/investor_sentiment_data_processed.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig")
print("原始数据行数：", len(df))

# 强制抽样 2000 行
df = df.sample(n=2000, random_state=42).reset_index(drop=True)
print("抽样后行数：", len(df))

# ---------------------- 2. 准备文本（只留短词） ----------------------
text_cols = ["content_processed", "comment1_processed", "comment2_processed"]
documents = []

for i, row in df.iterrows():
    words = []
    for col in text_cols:
        val = row[col]
        if pd.notna(val) and str(val).strip() != "":
            tokens = str(val).strip().split()
            # 只留 2~5 字的词，过滤垃圾
            tokens = [w for w in tokens if 2 <= len(w) <= 5]
            words.extend(tokens)
    if len(words) >= 3:
        documents.append(words)

print("有效文档数：", len(documents))
if len(documents) == 0:
    raise Exception("没有有效文本，检查你的分词列！")

# ---------------------- 3. 构建词典（强过滤，压到 2000 词） ----------------------
dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes(
    no_below=5,   # 至少出现5次
    no_above=0.7, # 不超过70%文档
    keep_n=2000   # 最多保留2000词
)
corpus = [dictionary.doc2bow(doc) for doc in documents]
print("词典大小：", len(dictionary))
print("语料构建完成")

# ---------------------- 4. 直接设定 K=5，不搜索最优（最省内存） ----------------------
K = 5
print(f"开始训练 LDA K={K} ...")

lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=K,
    random_state=42,
    passes=2,          # 极度减少迭代
    iterations=30,      # 很少迭代
    chunksize=100,      # 小批次
    update_every=1
)

print("训练完成！")

# 打印主题关键词
for idx, topic in lda_model.print_topics(num_words=8):
    print(f"主题{idx}：{topic}")

# ---------------------- 5. 简单统计主题分布 ----------------------
topic_probs = []
for bow in corpus:
    tp = lda_model.get_document_topics(bow, minimum_probability=0)
    topic_probs.append([p for _, p in tp])

avg_topic = np.mean(topic_probs, axis=0)
print("\n各主题平均占比：")
for i, p in enumerate(avg_topic):
    print(f"主题{i}：{p*100:.2f}%")

# 保存模型
lda_model.save("lda_simple.model")
print("\n模型已保存：lda_simple.model")

from gensim import models
# 加载模型
lda_model = models.LdaModel.load("lda_simple.model")
# 打印主题词
for idx, topic in lda_model.print_topics(num_words=8):
    print(f"主题{idx}：{topic}")

    # 把这段代码加在你原来的极简版代码的最后面

    import matplotlib.pyplot as plt

    # 从控制台复制你自己的主题占比数据
    topic_names = [
        "新能源技术与基本面",
        "市场回顾与交易行为",
        "机构动向与长期趋势",
        "储能业务与市场竞争",
        "白酒行业与消费属性"
    ]
    topic_ratios = [24.70, 15.56, 17.44, 25.68, 16.61]

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    bars = plt.bar(topic_names, topic_ratios, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])

    # 加上数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.2f}%", ha="center")

    plt.title("投资者文本LDA主题分布柱状图", fontsize=14)
    plt.ylabel("占比（%）")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("LDA主题分布柱状图.png", dpi=300)
    plt.show()
    print("✅ 已生成：LDA主题分布柱状图.png")