import warnings

warnings.filterwarnings("ignore")

# ========================= 1. 核心库导入（全版本兼容） =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 数据预处理与评估
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# LSTM模型构建
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 随机种子固定，保证结果100%可复现
np.random.seed(42)
import tensorflow as tf

tf.random.set_seed(42)

# ========================= 2. 全局配置（与论文完全对齐） =========================
# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 图片/结果保存路径（和你之前的代码完全一致，当前项目目录）
SAVE_PATH = "./"
# 模型超参数（与论文章节完全对齐）
TIME_STEP = 5  # 时间步长5，对应连续5个交易日
TRAIN_RATIO = 0.8  # 训练集占比80%
BATCH_SIZE = 32  # 批大小
EPOCHS = 100  # 最大迭代轮数
PATIENCE = 10  # 早停耐心值
LSTM_UNITS_1 = 64  # 第一层LSTM神经元数
LSTM_UNITS_2 = 32  # 第二层LSTM神经元数
DENSE_UNITS = 16  # 全连接层神经元数
DROPOUT_RATE = 0.2  # Dropout失活率
LEARNING_RATE = 0.001  # 初始学习率


# ========================= 3. 核心函数封装（彻底解决NameError报错） =========================
# 【核心修复】通用LSTM模型构建函数，全样本/分标均可调用，彻底解决未定义问题
def build_lstm_model(input_shape):
    """
    通用LSTM模型构建函数，与论文网络结构100%匹配
    :param input_shape: 输入形状 (时间步长, 特征数)
    :return: 编译后的LSTM模型
    """
    model = Sequential([
        # 第一层LSTM：64个隐藏单元，输出完整时序特征
        LSTM(
            LSTM_UNITS_1,
            activation='relu',
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.L2(0.001)
        ),
        Dropout(DROPOUT_RATE),
        # 第二层LSTM：32个隐藏单元，输出最终时序特征
        LSTM(
            LSTM_UNITS_2,
            activation='relu',
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.L2(0.001)
        ),
        Dropout(DROPOUT_RATE),
        # 全连接层：16个神经元，ReLU激活
        Dense(DENSE_UNITS, activation='relu'),
        # 输出层：1个神经元，线性激活（回归任务）
        Dense(1, activation='linear')
    ])

    # 模型编译
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


# 时序数据构建函数（适配LSTM三维输入）
def build_time_series_data(data, features, target_col, time_step=TIME_STEP):
    """
    构建LSTM所需的三维时序样本
    :param data: 原始数据框
    :param features: 输入特征列表
    :param target_col: 预测标签列名
    :param time_step: 时间步长
    :return: X_seq (样本数, 时间步长, 特征数), y_seq (样本数, 1)
    """
    X = data[features].values
    y = data[target_col].values

    X_seq, y_seq = [], []
    for i in range(time_step, len(X)):
        X_seq.append(X[i - time_step:i, :])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


# 核心评估指标计算函数
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# 单标的模型训练与评估函数
def train_evaluate_stock_model(stock_df, stock_name):
    """单标的模型训练与评估函数，调用通用build_lstm_model，彻底解决未定义问题"""
    print(f"\n📊 {stock_name} LSTM模型训练与评估")
    print("-" * 60)

    # 构建时序数据
    X_seq, y_seq = build_time_series_data(stock_df, features, "next_return")
    if len(X_seq) < 10:
        print(f"⚠️  {stock_name} 样本量不足，跳过建模")
        return None, None, None

    # 时间序列划分
    split_idx = int(len(X_seq) * TRAIN_RATIO)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # 标准化
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_flat = X_test.reshape(-1, X_test.shape[2])
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    # 【核心修复】调用通用的build_lstm_model函数，不再重复构建
    input_shape = (TIME_STEP, len(features))
    model = build_lstm_model(input_shape)
    # 早停机制
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)
    # 模型训练
    model.fit(
        X_train_scaled, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )

    # 预测与评估
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    rmse, mae, r2 = calculate_metrics(y_test, y_pred)
    # 方向准确率
    y_test_dir = np.where(y_test > 0, 1, 0)
    y_pred_dir = np.where(y_pred > 0, 1, 0)
    dir_acc = np.mean(y_test_dir == y_pred_dir)

    print(f"RMSE：{rmse:.4f}")
    print(f"MAE：{mae:.4f}")
    print(f"R²：{r2:.4f}")
    print(f"方向准确率：{dir_acc:.2%}")
    print("-" * 60)

    return model, scaler, (rmse, mae, r2, dir_acc)


# LSTM模型排列重要性计算函数（全版本兼容）
def calculate_lstm_permutation_importance(model, X_test, y_test, features, time_step, n_repeats=10, random_state=42):
    """
    自定义LSTM模型排列重要性计算，适配三维输入，全Python/sklearn版本兼容
    :param model: 训练好的LSTM模型
    :param X_test: 测试集三维输入
    :param y_test: 测试集标签
    :param features: 特征名称列表
    :param time_step: 时间步长
    :param n_repeats: 排列重复次数
    :param random_state: 随机种子
    :return: 特征重要性DataFrame
    """
    np.random.seed(random_state)
    # 计算原始模型基准MSE
    y_pred_original = model.predict(X_test, verbose=0).flatten()
    baseline_mse = mean_squared_error(y_test, y_pred_original)

    # 初始化特征重要性数组
    feature_importance = np.zeros(len(features))
    feature_importance_std = np.zeros(len(features))

    # 对每个特征进行排列扰动
    for feature_idx in range(len(features)):
        mse_repeats = []
        for _ in range(n_repeats):
            # 复制测试集数据
            X_permuted = X_test.copy()
            # 对当前特征在所有时间步的数值进行随机排列
            for t in range(time_step):
                X_permuted[:, t, feature_idx] = np.random.permutation(X_permuted[:, t, feature_idx])
            # 计算排列后的MSE
            y_pred_permuted = model.predict(X_permuted, verbose=0).flatten()
            permuted_mse = mean_squared_error(y_test, y_pred_permuted)
            mse_repeats.append(permuted_mse)

        # 计算该特征的重要性：排列后MSE与基准MSE的差值均值
        feature_importance[feature_idx] = np.mean(mse_repeats) - baseline_mse
        feature_importance_std[feature_idx] = np.std(mse_repeats)

    # 构建结果DataFrame
    importance_df = pd.DataFrame({
        "特征名称": features,
        "特征重要性": feature_importance,
        "标准差": feature_importance_std
    }).sort_values("特征重要性", ascending=False).reset_index(drop=True)
    importance_df["排名"] = importance_df.index + 1
    # 归一化为百分比，符合论文表述规范
    importance_df["特征重要性"] = (importance_df["特征重要性"] / importance_df["特征重要性"].sum() * 100).round(2)
    importance_df = importance_df[["特征名称", "特征重要性", "排名", "标准差"]]

    return importance_df


# ========================= 4. 数据加载与预处理（和随机森林模型完全对齐） =========================
# 4.1 加载原始数据（和你之前的文件名完全一致）
investor_df = pd.read_csv("data/投资者情绪_最终特征表.csv", encoding="utf-8-sig")
price_df = pd.read_csv("data/public_stock_market_data_final.csv", encoding="utf-8-sig")

# 4.2 日期格式统一，确保合并精准
investor_df["date"] = pd.to_datetime(investor_df["date"])
price_df["date"] = pd.to_datetime(price_df["date"])

# 4.3 按日期合并数据，按标的+日期排序
df = pd.merge(price_df, investor_df, on="date", how="inner")
df["stock_code"] = df["stock_code"].astype(str).str.strip()
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 4.4 生成预测标签（次日收益率，和随机森林完全一致）
df["next_return"] = df.groupby("stock_code")["log_return"].shift(-1)
# 剔除标签缺失值
df = df.dropna(subset=["next_return"])

# 4.5 极端值缩尾处理，剔除涨跌停干扰
df["next_return"] = df["next_return"].clip(lower=-0.09, upper=0.09)
df["log_return"] = df["log_return"].clip(lower=-0.09, upper=0.09)

# 4.6 特征工程（和随机森林模型的17维特征集完全一致，确保对比公平）
base_features = [
    "investor_mean",
    "investor_std",
    "investor_extreme",
    "investor_pos_ratio",
    "investor_count",
    "investor_lag1",
    "investor_lag3",
    "investor_lag5"
]
# 衍生特征
for lag in [1, 3, 5]:
    df[f"return_lag{lag}"] = df.groupby("stock_code")["log_return"].shift(lag)
df["mean_return_interact"] = df["investor_mean"] * df["return_lag1"]
df["std_return_interact"] = df["investor_std"] * df["return_lag1"]
df["mean_roll3"] = df.groupby("stock_code")["investor_mean"].transform(lambda x: x.rolling(3).mean())
df["std_roll3"] = df.groupby("stock_code")["investor_std"].transform(lambda x: x.rolling(3).std())
df["return_vol_5"] = df.groupby("stock_code")["log_return"].transform(lambda x: x.rolling(5).std())

# 最终特征集（17维，和论文完全一致）
features = base_features + [
    "return_lag1", "return_lag3", "return_lag5",
    "mean_return_interact", "std_return_interact",
    "mean_roll3", "std_roll3", "return_vol_5"
]

# 剔除特征缺失值
df = df.dropna(subset=features)
# 按标的拆分数据（用于分标的建模）
df_maotai = df[df["stock_code"].str.contains("600519")].reset_index(drop=True)
df_ningde = df[df["stock_code"].str.contains("300750")].reset_index(drop=True)

# 输出样本统计（和论文一致）
print("=" * 60)
print("📊 LSTM建模样本统计")
print("=" * 60)
print(f"总有效样本数：{len(df)} 条")
print(f"贵州茅台样本数：{len(df_maotai)} 条")
print(f"宁德时代样本数：{len(df_ningde)} 条")
print("=" * 60 + "\n")

# ========================= 5. 全样本LSTM模型训练与评估 =========================
# 构建全样本时序数据
X_seq, y_seq = build_time_series_data(df, features, "next_return")

# 时间序列划分（严格按时间前80%训练，后20%测试，无随机打乱，符合金融时序规范）
split_idx = int(len(X_seq) * TRAIN_RATIO)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# 数据标准化（仅在训练集拟合，避免数据泄露，符合学术规范）
scaler = StandardScaler()
# 展平时序数据进行标准化，再恢复三维形状
X_train_flat = X_train.reshape(-1, X_train.shape[2])
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
X_test_flat = X_test.reshape(-1, X_test.shape[2])
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# 输出数据集形状
print("=" * 60)
print("📊 时序数据集形状")
print("=" * 60)
print(f"训练集输入形状：{X_train_scaled.shape} (样本数, 时间步长, 特征数)")
print(f"测试集输入形状：{X_test_scaled.shape} (样本数, 时间步长, 特征数)")
print(f"训练集标签形状：{y_train.shape}")
print(f"测试集标签形状：{y_test.shape}")
print("=" * 60 + "\n")

# 【核心修复】调用通用的build_lstm_model函数构建全样本模型
input_shape = (TIME_STEP, len(features))
model = build_lstm_model(input_shape)

# 输出模型结构（和论文一致）
print("=" * 60)
print("📊 LSTM模型结构")
print("=" * 60)
model.summary()
print("=" * 60 + "\n")

# 模型训练
print("=" * 80)
print("🚀 开始全样本LSTM模型训练")
print("=" * 80)
# 早停机制设置
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)
# 模型训练
history = model.fit(
    X_train_scaled, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)
print("=" * 80)
print("✅ 全样本模型训练完成！")
print("=" * 80 + "\n")

# 全样本模型评估
# 模型预测
y_pred = model.predict(X_test_scaled, verbose=0).flatten()
y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()

# 训练集与测试集指标计算
train_rmse, train_mae, train_r2 = calculate_metrics(y_train, y_train_pred)
test_rmse, test_mae, test_r2 = calculate_metrics(y_test, y_pred)

# 涨跌方向准确率计算
y_test_direction = np.where(y_test > 0, 1, 0)
y_pred_direction = np.where(y_pred > 0, 1, 0)
direction_accuracy = np.mean(y_test_direction == y_pred_direction)

# 输出全样本评估结果（和论文表格完全一致）
print("=" * 60)
print("📈 全样本LSTM模型评估结果（论文核心指标）")
print("=" * 60)
print(f"【训练集拟合效果】")
print(f"RMSE：{train_rmse:.4f}")
print(f"MAE：{train_mae:.4f}")
print(f"R²：{train_r2:.4f}")
print(f"\n【测试集泛化效果】")
print(f"RMSE：{test_rmse:.4f}")
print(f"MAE：{test_mae:.4f}")
print(f"R²：{test_r2:.4f}")
print(f"方向准确率：{direction_accuracy:.2%}")
print("=" * 60 + "\n")

# ========================= 6. 分标的模型训练与评估（贵州茅台+宁德时代） =========================
# 贵州茅台模型
model_maotai, scaler_maotai, metrics_maotai = train_evaluate_stock_model(df_maotai, "贵州茅台")
# 宁德时代模型
model_ningde, scaler_ningde, metrics_ningde = train_evaluate_stock_model(df_ningde, "宁德时代")

# ========================= 7. 特征重要性分析（全版本兼容，和论文完全一致） =========================
print("\n" + "=" * 80)
print("🔍 特征重要性分析（排列重要性法，符合学术规范）")
print("=" * 80)

# 调用函数计算特征重要性
importance_df = calculate_lstm_permutation_importance(
    model=model,
    X_test=X_test_scaled,
    y_test=y_test,
    features=features,
    time_step=TIME_STEP,
    n_repeats=10,
    random_state=42
)

# 输出特征重要性排名（和论文表格完全一致）
print("📋 特征重要性排名（归一化后）")
print(importance_df.to_string(index=False))
print("=" * 80)

# 输出TOP3核心特征（和论文一致）
print("\n📌 核心特征TOP3")
print("=" * 60)
for i in range(3):
    print(
        f"第{importance_df.iloc[i]['排名']}名：{importance_df.iloc[i]['特征名称']}，重要性占比{importance_df.iloc[i]['特征重要性']}%")
print("=" * 60 + "\n")

# ========================= 8. 可视化图表生成（和论文图表完全一致） =========================
print("=" * 80)
print("🖼️  论文可视化图表生成")
print("=" * 80)

# 8.1 模型训练损失曲线（图5.4.3-2）
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='训练损失', color="#4472C4", linewidth=1.5)
plt.plot(history.history['val_loss'], label='验证损失', color="#C00000", linewidth=1.5)
plt.title("LSTM模型训练损失曲线", fontsize=14, fontweight="bold")
plt.xlabel("迭代轮次", fontsize=12)
plt.ylabel("MSE损失", fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}LSTM_训练损失曲线.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ 已生成：LSTM_训练损失曲线.png（图5.4.3-2）")

# 8.2 预测值与真实值对比图（图5.4.3-3）
plt.figure(figsize=(14, 6))
plot_len = min(150, len(y_test))
plt.plot(range(plot_len), y_test[:plot_len], label="真实收益率", color="#4472C4", linewidth=1.5)
plt.plot(range(plot_len), y_pred[:plot_len], label="预测收益率", color="#C00000", linestyle="--", linewidth=1.5)
plt.title("LSTM模型预测收益与真实收益对比图", fontsize=14, fontweight="bold")
plt.xlabel("测试集样本序号", fontsize=12)
plt.ylabel("股票对数收益率", fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}LSTM_预测真实对比图.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ 已生成：LSTM_预测真实对比图.png（图5.4.3-3）")

# 8.3 特征重要性排序图（图5.4.3-4）
plt.figure(figsize=(12, 6))
plt.barh(importance_df["特征名称"], importance_df["特征重要性"], color="#4472C4")
plt.xlabel("特征重要性（%）", fontsize=12)
plt.title("投资者情绪LSTM模型特征重要性排序", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}LSTM_特征重要性图.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ 已生成：LSTM_特征重要性图.png（图5.4.3-4）")

print("\n🎉 所有论文可视化图表生成完成！可直接插入论文！")
print("=" * 80)

# ========================= 9. 最终结果汇总（论文可直接复制） =========================
print("\n" + "=" * 100)
print("📋 5.4.3 LSTM预测分析 核心结果汇总（论文可直接复制）")
print("=" * 100)
print(f"1. 全样本模型核心指标：")
print(f"   测试集RMSE：{test_rmse:.4f}，MAE：{test_mae:.4f}，R²：{test_r2:.4f}，方向准确率：{direction_accuracy:.2%}")
print(f"2. 分标的模型核心指标：")
if metrics_maotai:
    print(
        f"   贵州茅台：RMSE：{metrics_maotai[0]:.4f}，MAE：{metrics_maotai[1]:.4f}，R²：{metrics_maotai[2]:.4f}，方向准确率：{metrics_maotai[3]:.2%}")
if metrics_ningde:
    print(
        f"   宁德时代：RMSE：{metrics_ningde[0]:.4f}，MAE：{metrics_ningde[1]:.4f}，R²：{metrics_ningde[2]:.4f}，方向准确率：{metrics_ningde[3]:.2%}")
print(f"3. 核心特征排名TOP3：")
for i in range(3):
    print(
        f"   第{importance_df.iloc[i]['排名']}名：{importance_df.iloc[i]['特征名称']}，重要性占比{importance_df.iloc[i]['特征重要性']}%")
print("=" * 100)
print("\n🎉 5.4.3 LSTM预测分析 全流程运行完成！所有结果可直接写入论文！")