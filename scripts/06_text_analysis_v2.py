import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings

warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows系统
# plt.rcParams["font.sans-serif"] = ["PingFang SC"]  # Mac系统
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = (10, 6)


# ===================== 1. 修复后的词频统计函数 =====================
def calculate_word_frequency(df, text_columns=["content_processed", "comment1_processed", "comment2_processed"],
                             stock_code=None):
    """
    修复：增加stock_code类型兼容、空值检查、调试信息输出
    """
    all_words = []

    if stock_code is not None:
        # 关键修复1：同时支持数字和字符串类型的stock_code
        df_stock = df[
            (df["stock_code"] == stock_code) |
            (df["stock_code"] == str(stock_code)) |
            (df["stock_code"] == int(stock_code))
            ].reset_index(drop=True)

        print(f"🔍 筛选股票 {stock_code}，匹配到 {len(df_stock)} 条评论")

        if len(df_stock) == 0:
            print(f"⚠️  警告：未找到股票 {stock_code} 的任何评论数据")
            return Counter(), 0, 0
    else:
        df_stock = df

    for col in text_columns:
        for text in df_stock[col].dropna():
            if str(text).strip() != "":
                words = str(text).strip().split(" ")
                valid_words = [word for word in words if len(word) > 1 and word.strip() != ""]
                all_words.extend(valid_words)

    counter = Counter(all_words)
    return counter, len(all_words), len(set(all_words))


# ===================== 2. 原有图表函数（保留） =====================
def generate_overall_wordcloud(word_counter, save_path="1_整体投资者词云图.png"):
    if len(word_counter) == 0:
        print(f"⚠️  跳过 {save_path}：无有效词频数据")
        return

    filtered_words = {word: count for word, count in word_counter.items() if count >= 50}
    if len(filtered_words) == 0:
        filtered_words = dict(word_counter.most_common(100))

    wc = WordCloud(
        font_path="C:/Windows/Fonts/simhei.ttf",
        # font_path="/System/Library/Fonts/PingFang.ttc",  # Mac系统
        width=1920, height=1080, background_color="white",
        max_words=300, max_font_size=250, collocations=False
    )
    wc.generate_from_frequencies(filtered_words)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("东方财富股吧整体投资者评论词云图", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ 已生成：{save_path}")


def plot_top_words_bar(word_counter, save_path="2_Top20整体高频词柱状图.png"):
    if len(word_counter) == 0:
        print(f"⚠️  跳过 {save_path}：无有效词频数据")
        return

    top_words = word_counter.most_common(20)
    words, counts = zip(*top_words)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(words[::-1], counts[::-1], color="#1f77b4", alpha=0.8)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 80, bar.get_y() + bar.get_height() / 2, f"{int(width)}", ha="left", va="center", fontsize=10)

    plt.xlabel("出现频次", fontsize=12)
    plt.ylabel("词语", fontsize=12)
    plt.title("投资者评论Top20高频词柱状图", fontsize=14, pad=20)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ 已生成：{save_path}")


# ===================== 3. 修复后的分个股对比词云图 =====================
def generate_stock_comparison_wordcloud(df, save_path="3_分个股对比词云图.png"):
    """
    修复：降低词频阈值、增加空值检查、异常处理
    """
    try:
        counter_600519, _, _ = calculate_word_frequency(df, stock_code=600519)
        counter_300750, _, _ = calculate_word_frequency(df, stock_code=300750)

        # 关键修复2：降低词频阈值从30到10，自动适配数据量
        filtered_600519 = {word: count for word, count in counter_600519.items() if count >= 10}
        filtered_300750 = {word: count for word, count in counter_300750.items() if count >= 10}

        # 关键修复3：如果过滤后为空，自动取Top100词
        if len(filtered_600519) == 0:
            print("⚠️  贵州茅台词频过低，自动取Top100词生成词云")
            filtered_600519 = dict(counter_600519.most_common(100))
        if len(filtered_300750) == 0:
            print("⚠️  宁德时代词频过低，自动取Top100词生成词云")
            filtered_300750 = dict(counter_300750.most_common(100))

        wc_600519 = WordCloud(
            font_path="C:/Windows/Fonts/simhei.ttf",
            width=900, height=600, background_color="#f8f9fa",
            max_words=150, max_font_size=150, collocations=False
        ).generate_from_frequencies(filtered_600519)

        wc_300750 = WordCloud(
            font_path="C:/Windows/Fonts/simhei.ttf",
            width=900, height=600, background_color="#f8f9fa",
            max_words=150, max_font_size=150, collocations=False
        ).generate_from_frequencies(filtered_300750)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(wc_600519, interpolation="bilinear")
        ax1.set_title("贵州茅台（600519）投资者词云图", fontsize=14, pad=15)
        ax1.axis("off")

        ax2.imshow(wc_300750, interpolation="bilinear")
        ax2.set_title("宁德时代（300750）投资者词云图", fontsize=14, pad=15)
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✅ 已生成：{save_path}")

    except Exception as e:
        print(f"❌ 生成分个股词云图失败：{str(e)}")
        print("💡 已跳过此图表，继续生成其他图表")


# ===================== 4. 其他新增图表函数（保留并增加异常处理） =====================
def plot_sentiment_word_comparison(word_counter, save_path="4_核心情绪词频对比图.png"):
    if len(word_counter) == 0:
        print(f"⚠️  跳过 {save_path}：无有效词频数据")
        return

    positive_words = ["上涨", "抄底", "加仓", "吃肉", "牛逼", "利好", "涨停", "看好"]
    negative_words = ["下跌", "卖出", "吃面", "垃圾", "利空", "跌停", "套牢", "踏空"]

    pos_counts = [word_counter.get(word, 0) for word in positive_words]
    neg_counts = [word_counter.get(word, 0) for word in negative_words]

    x = range(len(positive_words))
    width = 0.35

    plt.figure(figsize=(12, 7))
    bars1 = plt.bar([i - width / 2 for i in x], pos_counts, width, label="正向情绪词", color="#2ecc71", alpha=0.8)
    bars2 = plt.bar([i + width / 2 for i in x], neg_counts, width, label="负向情绪词", color="#e74c3c", alpha=0.8)

    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 30, f"{int(height)}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 30, f"{int(height)}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, positive_words, fontsize=11)
    plt.xlabel("核心情绪词", fontsize=12)
    plt.ylabel("出现频次", fontsize=12)
    plt.title("投资者核心情绪词频对比图", fontsize=14, pad=20)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ 已生成：{save_path}")


def plot_trading_behavior_pie(word_counter, save_path="5_核心交易行为占比图.png"):
    if len(word_counter) == 0:
        print(f"⚠️  跳过 {save_path}：无有效词频数据")
        return

    trading_words = {
        "抄底": word_counter.get("抄底", 0),
        "加仓": word_counter.get("加仓", 0),
        "卖出": word_counter.get("卖出", 0),
        "清仓": word_counter.get("清仓", 0),
        "持有": word_counter.get("持有", 0),
        "观望": word_counter.get("观望", 0)
    }

    # 过滤掉频次为0的类别
    trading_words = {k: v for k, v in trading_words.items() if v > 0}

    if len(trading_words) == 0:
        print(f"⚠️  跳过 {save_path}：无有效交易行为词")
        return

    labels = list(trading_words.keys())
    sizes = list(trading_words.values())
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f1c40f", "#95a5a6"][:len(labels)]
    explode = (0.05,) * len(labels)

    plt.figure(figsize=(10, 10))
    wedges, texts, autotexts = plt.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", shadow=True, startangle=90, textprops={"fontsize": 12}
    )
    plt.setp(autotexts, size=11, weight="bold")
    plt.title("投资者核心交易行为占比图", fontsize=16, pad=20)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ 已生成：{save_path}")


def plot_word_length_distribution(df, save_path="6_词汇长度分布直方图.png"):
    all_lengths = []
    text_columns = ["content_processed", "comment1_processed", "comment2_processed"]

    for col in text_columns:
        for text in df[col].dropna():
            if str(text).strip() != "":
                words = str(text).strip().split(" ")
                lengths = [len(word) for word in words if len(word) > 1 and word.strip() != ""]
                all_lengths.extend(lengths)

    if len(all_lengths) == 0:
        print(f"⚠️  跳过 {save_path}：无有效词汇数据")
        return

    plt.figure(figsize=(12, 7))
    plt.hist(all_lengths, bins=range(2, 11), color="#3498db", alpha=0.7, edgecolor="black", linewidth=1)

    plt.xlabel("词汇长度（字符数）", fontsize=12)
    plt.ylabel("出现频次", fontsize=12)
    plt.title("投资者评论文本词汇长度分布直方图", fontsize=14, pad=20)
    plt.xticks(range(2, 11))
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    mean_length = sum(all_lengths) / len(all_lengths)
    plt.axvline(mean_length, color="red", linestyle="--", linewidth=2, label=f"平均长度：{mean_length:.2f}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ 已生成：{save_path}")


def plot_stock_top_words_comparison(df, save_path="7_个股Top10高频词对比图.png"):
    try:
        counter_600519, _, _ = calculate_word_frequency(df, stock_code=600519)
        counter_300750, _, _ = calculate_word_frequency(df, stock_code=300750)

        top10_600519 = counter_600519.most_common(10)
        top10_300750 = counter_300750.most_common(10)

        if len(top10_600519) == 0 or len(top10_300750) == 0:
            print(f"⚠️  跳过 {save_path}：某只股票无有效高频词")
            return

        words_600519, counts_600519 = zip(*top10_600519)
        words_300750, counts_300750 = zip(*top10_300750)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        bars1 = ax1.barh(words_600519[::-1], counts_600519[::-1], color="#e67e22", alpha=0.8)
        ax1.set_title("贵州茅台Top10高频词", fontsize=14, pad=15)
        ax1.set_xlabel("出现频次", fontsize=12)
        ax1.grid(axis="x", linestyle="--", alpha=0.3)
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 30, bar.get_y() + bar.get_height() / 2, f"{int(width)}", ha="left", va="center",
                     fontsize=10)

        bars2 = ax2.barh(words_300750[::-1], counts_300750[::-1], color="#3498db", alpha=0.8)
        ax2.set_title("宁德时代Top10高频词", fontsize=14, pad=15)
        ax2.set_xlabel("出现频次", fontsize=12)
        ax2.grid(axis="x", linestyle="--", alpha=0.3)
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 30, bar.get_y() + bar.get_height() / 2, f"{int(width)}", ha="left", va="center",
                     fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✅ 已生成：{save_path}")

    except Exception as e:
        print(f"❌ 生成个股高频词对比图失败：{str(e)}")
        print("💡 已跳过此图表，继续生成其他图表")


# ===================== 主程序 =====================
if __name__ == "__main__":
    data_path = "data/investor_sentiment_data_processed.csv"
    try:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        print(f"✅ 数据读取成功，共{len(df)}条有效评论")
        print(f"📋 数据包含字段：{df.columns.tolist()}")
        print(f"📊 股票代码分布：\n{df['stock_code'].value_counts()}")

    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {data_path}")
        exit()
    except KeyError as e:
        print(f"❌ 错误：数据中缺少字段 {e}")
        print("💡 请确保CSV文件包含 stock_code、content_processed 等字段")
        exit()

    # 计算整体词频
    overall_counter, total_words, unique_words = calculate_word_frequency(df)
    print(f"\n📊 整体词频统计：总词数={total_words}，唯一词数={unique_words}")

    # 生成所有图表
    print("\n🚀 开始生成全部可视化图表...")
    generate_overall_wordcloud(overall_counter)
    plot_top_words_bar(overall_counter)
    generate_stock_comparison_wordcloud(df)
    plot_sentiment_word_comparison(overall_counter)
    plot_trading_behavior_pie(overall_counter)
    plot_word_length_distribution(df)
    plot_stock_top_words_comparison(df)

    print("\n🎉 全部可生成的图表已完成！")
    print("📋 生成的文件列表：")
    print("1. 1_整体投资者词云图.png")
    print("2. 2_Top20整体高频词柱状图.png")
    print("3. 3_分个股对比词云图.png")
    print("4. 4_核心情绪词频对比图.png")
    print("5. 5_核心交易行为占比图.png")
    print("6. 6_词汇长度分布直方图.png")
    print("7. 7_个股Top10高频词对比图.png")