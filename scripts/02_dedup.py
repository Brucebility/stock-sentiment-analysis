import jieba
import pandas as pd
import warnings

# 屏蔽兼容警告，不影响核心逻辑
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 中文停用词表（适配股吧文本）
stopwords = set([
    "的", "了", "和", "是", "我", "有", "在", "也", "就", "都", "不", "啊", "呀", "吧", "吗", "呢",
    "你", "他", "她", "它", "们", "这", "那", "个", "之", "以", "而", "于", "上", "中", "下", "为", "与",
])


def text_tokenize(text):
    """中文文本分词函数，过滤停用词与空字符串"""
    if pd.isna(text) or str(text).strip() == "":
        return []
    words = jieba.lcut(str(text).strip())
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    return filtered_words


def jaccard_similarity(text1, text2):
    """计算两条文本的Jaccard相似度系数"""
    words1 = set(text_tokenize(text1))
    words2 = set(text_tokenize(text2))
    intersection = words1 & words2
    union = words1 | words2
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)


def deduplicate_guba_comments(df, date_col="date", similarity_threshold=0.9):
    """
    股吧评论双层去重函数（修复版，适配5个核心字段的数据集）
    第一层：精确匹配去重
    第二层：Jaccard相似度近似去重
    """
    # 核心修复：自动合并主帖+回复内容，生成去重用的全量文本列
    df["full_content"] = df.apply(
        lambda row: "；".join([
            str(row["content"]) if pd.notna(row["content"]) else "",
            str(row["comment1"]) if pd.notna(row["comment1"]) else "",
            str(row["comment2"]) if pd.notna(row["comment2"]) else ""
        ]).strip("；"),
        axis=1
    )
    # 去除空文本行
    df_valid = df[df["full_content"] != ""].reset_index(drop=True)
    print(f"去重前有效数据总行数：{len(df_valid)}")

    # 第一层：精确匹配去重，按发布时间保留最早的一条
    df_exact_dedup = df_valid.sort_values(date_col).drop_duplicates(subset=["full_content"], keep="first").reset_index(
        drop=True)
    print(f"精确匹配去重后数据行数：{len(df_exact_dedup)}")
    print(f"去除完全重复内容数量：{len(df_valid) - len(df_exact_dedup)}")

    # 第二层：Jaccard相似度近似去重
    dedup_result = [df_exact_dedup.iloc[0]]
    for idx, row in df_exact_dedup.iloc[1:].iterrows():
        current_text = row["full_content"]
        is_duplicate = False
        # 与已保留的所有内容计算相似度
        for kept_row in dedup_result:
            kept_text = kept_row["full_content"]
            sim = jaccard_similarity(current_text, kept_text)
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            dedup_result.append(row)

    # 转换为DataFrame，仅保留你需要的5个核心字段
    df_final_dedup = pd.DataFrame(dedup_result)[["stock_code", "date", "content", "comment1", "comment2"]].reset_index(
        drop=True)
    print(f"近似匹配去重后最终数据行数：{len(df_final_dedup)}")
    print(f"去除近似重复内容数量：{len(df_exact_dedup) - len(df_final_dedup)}")
    print(f"总去重率：{(len(df_valid) - len(df_final_dedup)) / len(df_valid):.2%}")

    return df_final_dedup


# 批量执行去重
if __name__ == "__main__":
    # 配置文件路径（替换为你的清洗后CSV文件实际路径）
    cleaned_data_path = "data/investor_sentiment_data_cleaned.csv"
    final_dedup_data_path = "data/investor_sentiment_data_final.csv"

    # 读取清洗后的数据集
    df_cleaned = pd.read_csv(cleaned_data_path, encoding="utf-8-sig")
    print(f"✅ 清洗后数据集读取成功，共{len(df_cleaned)}行数据")
    print(f"数据集包含字段：{df_cleaned.columns.tolist()}")

    # 执行去重
    df_final = deduplicate_guba_comments(df_cleaned)

    # 保存最终结果
    df_final.to_csv(final_dedup_data_path, index=False, encoding="utf-8-sig")
    print(f"\n🎉 去重全流程完成！")
    print(f"📋 最终去重结果已保存至：{final_dedup_data_path}")
    print(f"📌 最终输出文件仅包含5个核心字段：stock_code、date、content、comment1、comment2")