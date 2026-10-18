import jieba
import pandas as pd
import warnings

# 屏蔽所有兼容警告，不影响核心逻辑
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===================== 1. 分词处理核心配置（股吧专属定制） =====================
# 1.1 股吧投资者专属自定义词典（解决专业词汇错误切分问题）
guba_custom_dict = [
    # 个股简称
    "宁王", "茅王", "宁德", "茅台", "招行", "平安", "比亚迪", "隆基",
    # 社区俚语
    "吃面", "吃肉", "踏空", "套牢", "解套", "回本", "绝绝子", "yyds",
    # 交易术语
    "主力", "游资", "散户", "机构", "北向", "南向", "抄底", "逃顶", "做T",
    "加仓", "减仓", "清仓", "补仓", "主力资金", "机构资金", "游资资金",
    # 情绪表达词
    "牛逼", "厉害", "强", "垃圾", "完蛋", "拉胯", "给力", "拉了"
]
# 批量加载自定义词典，确保专业词汇完整识别
for word in guba_custom_dict:
    jieba.add_word(word, freq=1000, tag="n")
print("✅ 股吧专属自定义词典加载完成，共", len(guba_custom_dict), "个核心词汇")

# 1.2 投资者场景专属停用词表（过滤低信息量词汇）
stopwords = set([
    # 通用虚词
    "的", "了", "在", "是", "和", "与", "但", "而", "也", "都", "这", "那", "有", "对", "从", "到",
    # 标点符号
    "，", "。", "！", "？", "；", "：", "“", "”", "‘", "’", "、", "…", "—",
    # 口语化语气词
    "哈哈", "呵呵", "嘿嘿", "唉", "哎", "哼", "切", "哦", "嗯", "哈", "嘿",
    # 无意义互动词
    "顶", "沙发", "楼主", "路过", "楼上", "楼下", "前排", "后排", "板凳", "地板",
    # 高频无区分度词
    "股票", "股价", "公司", "市场", "今天", "明天", "今日", "昨日", "宁王", "茅王"
])
print("✅ 停用词表加载完成，共", len(stopwords), "个停用词")

# 1.3 投资者文本同义词映射表（统一语义表述，减少特征冗余）
synonym_map = {
    # 股价变动类
    "上涨": ["涨", "走高", "攀升", "上扬", "反弹", "拉升", "起飞"],
    "下跌": ["跌", "走低", "回落", "下挫", "回调", "跳水", "下杀"],
    # 交易行为类
    "买入": ["买进", "购入", "建仓", "进场", "布局", "加仓", "补仓"],
    "卖出": ["抛售", "减持", "出局", "离场", "清仓", "减仓"],
    # 市场参与者类
    "机构": ["主力", "大户", "机构投资者", "基金", "主力资金"],
    # 公司财务类
    "营业收入": ["营收", "收入", "销售额"],
    # 情绪表达类
    "厉害": ["牛逼", "强", "给力", "绝绝子", "yyds"],
    "垃圾": ["拉胯", "完蛋", "拉了", "不行", "不好"]
}
print("✅ 同义词映射表加载完成，共", len(synonym_map), "组同义词")


# ===================== 2. 分词处理核心函数 =====================
def cut_text(text):
    """
    单条文本分词函数，采用jieba精确模式，适配股吧短句特征
    返回空格分隔的标准化词语序列
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    # 精确模式分词，无冗余切分，完全适配股吧短句
    words = jieba.lcut(str(text).strip(), cut_all=False)
    # 过滤空字符串
    filtered_words = [word for word in words if word.strip() != ""]
    # 空格分隔返回
    return " ".join(filtered_words)


def remove_stopwords(text):
    """
    去停用词函数，过滤低信息量词汇，压缩特征维度
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    # 按空格切分为词语列表
    words = str(text).strip().split(" ")
    # 过滤停用词和空字符串
    filtered_words = [word for word in words if word not in stopwords and word.strip() != ""]
    # 空格分隔返回
    return " ".join(filtered_words)


def merge_synonyms(text):
    """
    同义词合并函数，统一同一语义的多种口语化表述
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    result_text = str(text).strip()
    # 遍历同义词映射表，统一替换
    for standard_word, synonyms in synonym_map.items():
        for synonym in synonyms:
            # 替换中间的同义词
            result_text = result_text.replace(f" {synonym} ", f" {standard_word} ")
            # 替换开头的同义词
            if result_text.startswith(f"{synonym} "):
                result_text = result_text.replace(f"{synonym} ", f"{standard_word} ", 1)
            # 替换结尾的同义词
            if result_text.endswith(f" {synonym}"):
                result_text = result_text[:-len(f" {synonym}")] + f" {standard_word}"
    return result_text.strip()


def process_text(text):
    """
    全流程处理函数：分词→去停用词→同义词合并
    一键完成单条文本的标准化处理
    """
    cut_result = cut_text(text)
    stop_result = remove_stopwords(cut_result)
    merge_result = merge_synonyms(stop_result)
    return merge_result


# ===================== 3. 全量批量分词执行（统一使用cleaned.csv文件） =====================
if __name__ == "__main__":
    # 配置文件路径（统一使用清洗后的文件，无需去重文件）
    cleaned_data_path = "data/investor_sentiment_data_cleaned.csv"
    final_process_data_path = "data/investor_sentiment_data_processed.csv"

    # 读取清洗后的数据集
    try:
        df = pd.read_csv(cleaned_data_path, encoding="utf-8-sig")
        print(f"✅ 清洗后数据集读取成功，共{len(df)}行有效数据")
        print(f"数据集包含字段：{df.columns.tolist()}")
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {cleaned_data_path}")
        print("💡 解决方案：1. 核对文件路径是否正确；2. 确保文件已放在对应文件夹中")
        exit()

    # 全量分词处理：对主帖、一楼回复、二楼回复分别进行全流程标准化
    print("🔄 正在进行分词→去停用词→同义词合并全流程处理...")
    df["content_processed"] = df["content"].apply(process_text)
    df["comment1_processed"] = df["comment1"].apply(process_text)
    df["comment2_processed"] = df["comment2"].apply(process_text)
    print("✅ 全量分词处理完成")

    # 保存最终处理结果
    df.to_csv(final_process_data_path, index=False, encoding="utf-8-sig")
    print(f"✅ 最终处理结果已保存至：{final_process_data_path}")

    # 输出分词效果示例（前5条，直观展示处理效果）
    print("\n" + "=" * 100)
    print("🔍 分词处理效果示例（前5条）")
    print("=" * 100)
    for idx, row in df.head(5).iterrows():
        print(f"【行号{idx + 1} | 股票代码：{row['stock_code']} | 日期：{row['date']}】")
        print(f"原始主帖：{row['content'][:80]}")
        print(f"分词结果：{cut_text(row['content'])[:100]}")
        print(f"去停用词后：{remove_stopwords(cut_text(row['content']))[:100]}")
        print(f"同义词合并后：{row['content_processed'][:100]}")
        print("-" * 100)

    # 最终结果统计
    print("\n" + "=" * 50)
    print("📊 分词处理最终结果统计")
    print("=" * 50)
    print(f"处理前有效数据总行数：{len(df)}")
    print(f"主帖非空处理结果数量：{len(df[df['content_processed'] != ''])}")
    print(f"一楼回复非空处理结果数量：{len(df[df['comment1_processed'] != ''])}")
    print(f"二楼回复非空处理结果数量：{len(df[df['comment2_processed'] != ''])}")
    print("=" * 50)
    print("🎉 全流程分词标准化处理完成！")