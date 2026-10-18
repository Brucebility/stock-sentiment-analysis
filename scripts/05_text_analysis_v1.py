import pandas as pd
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings

warnings.filterwarnings("ignore")

# ===================== 全局配置（统一中文显示） =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows系统用SimHei
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120  # 图表清晰度


# ===================== 1. 词频统计核心函数 =====================
def calculate_word_frequency(df, text_columns=["content_processed", "comment1_processed", "comment2_processed"]):
    """
    计算投资者文本的词频，合并主帖、一楼回复、二楼回复的所有内容
    :param df: 预处理后的投资者数据DataFrame
    :param text_columns: 需要统计的文本字段列表
    :return: 词频计数器Counter对象，总词数，唯一词数
    """
    all_words = []

    # 遍历所有文本字段，合并所有词语
    for col in text_columns:
        for text in df[col].dropna():
            if str(text).strip() != "":
                words = str(text).strip().split(" ")
                # 过滤空字符串和单字符（进一步降噪）
                valid_words = [word for word in words if len(word) > 1 and word.strip() != ""]
                all_words.extend(valid_words)

    # 统计词频
    word_counter = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_counter)

    return word_counter, total_words, unique_words


# ===================== 2. 词云图绘制函数 =====================
def generate_wordcloud(word_counter, save_path="investor_wordcloud.png", min_frequency=50):
    """
    生成投资者评论词云图
    :param word_counter: 词频计数器
    :param save_path: 词云图保存路径
    :param min_frequency: 最小词频，过滤低频词
    """
    # 过滤低频词，提升词云可读性
    filtered_words = {word: count for word, count in word_counter.items() if count >= min_frequency}

    # 生成词云（Windows系统字体路径，Mac/Linux需修改为对应字体路径）
    wc = WordCloud(
        font_path="C:/Windows/Fonts/simhei.ttf",
        width=1600,
        height=900,
        background_color="white",
        max_words=200,
        max_font_size=200,
        collocations=False  # 避免重复显示词语
    )

    wc.generate_from_frequencies(filtered_words)

    # 绘制并保存词云图
    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("东方财富股吧投资者评论词云图", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"✅ 词云图已保存至：{save_path}")


# ===================== 3. 高频词柱状图绘制函数 =====================
def plot_top_words_bar(word_counter, top_n=20, save_path="investor_top_words_bar.png"):
    """
    绘制Top N高频词柱状图
    :param word_counter: 词频计数器
    :param top_n: 显示前N个高频词
    :param save_path: 柱状图保存路径
    """
    # 获取Top N高频词
    top_words = word_counter.most_common(top_n)
    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.barh(words[::-1], counts[::-1], color="#1f77b4", alpha=0.8)

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=10
        )

    # 图表美化
    plt.xlabel("出现频次", fontsize=12)
    plt.ylabel("词语", fontsize=12)
    plt.title(f"投资者评论Top {top_n} 高频词柱状图", fontsize=14, pad=20)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"✅ Top {top_n} 高频词柱状图已保存至：{save_path}")


# ===================== 4. 主程序执行 =====================
if __name__ == "__main__":
    # 读取预处理后的投资者数据（替换为你的实际文件路径）
    data_path = "data/investor_sentiment_data_processed.csv"
    try:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        print(f"✅ 数据读取成功，共{len(df)}条有效评论")
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {data_path}")
        print("💡 请检查文件路径是否正确，确保已完成分词预处理")
        exit()

    # 第一步：计算词频
    print("\n" + "=" * 50)
    print("📊 第一步：计算词频统计")
    print("=" * 50)
    word_counter, total_words, unique_words = calculate_word_frequency(df)
    print(f"总词数：{total_words}")
    print(f"唯一词数：{unique_words}")
    print(f"词汇丰富度(唯一词/总词)：{unique_words / total_words:.4f}")

    # 输出Top20高频词
    print("\nTop20高频词：")
    for i, (word, count) in enumerate(word_counter.most_common(20), 1):
        print(f"{i}. {word}: {count}次")

    # 第二步：生成词云图
    print("\n" + "=" * 50)
    print("☁️  第二步：生成词云图")
    print("=" * 50)
    generate_wordcloud(word_counter)

    # 第三步：绘制高频词柱状图
    print("\n" + "=" * 50)
    print("📈 第三步：绘制高频词柱状图")
    print("=" * 50)
    plot_top_words_bar(word_counter)

    print("\n" + "=" * 50)
    print("🎉 词频统计与可视化全部完成！")
    print("=" * 50)