import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Add, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.optimizers import Adam


# 1. 数据加载与预处理
def load_and_preprocess_data():
    folder_path = '/Users/xutong/Documents/风电功率预测/ResNet + Transformer 模型/data/raw'
    merged_file_path = os.path.join('/Users/xutong/Documents/风电功率预测/ResNet + Transformer 模型/data/user_data', '汇总.xlsx')
    # 读取已合并的文件
    df = pd.read_excel(merged_file_path, sheet_name="Sheet1")
    print(f"[DEBUG] 总样本数: {len(df)}")  # 验证数据量一致性

    # 处理时间戳
    df["时间"] = pd.to_datetime(df["年份"])
    df.set_index("时间", inplace=True)
    df["小时"] = df.index.hour
    df["星期几"] = df.index.dayofweek
    df["月份"] = df.index.month

    # 添加滞后特征
    lag_steps = 3
    for lag in range(1, lag_steps + 1):
        df[f"风速_lag{lag}"] = df["风速"].shift(lag)
        df[f"发电功率_lag{lag}"] = df["发电功率"].shift(lag)
    df.dropna(inplace=True)

    # 标准化
    features = ["空气密度", "风速", "小时", "星期几", "月份"] + [col for col in df.columns if "lag" in col]
    target = "发电功率"
    features_and_target = features + [target]
    scaler = StandardScaler()
    df[features_and_target] = scaler.fit_transform(df[features_and_target])

    # 分割数据集
    train_size = int(0.7 * len(df))
    val_size = int(0.2 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    return train_df, val_df, test_df, features, target, scaler


# 2. 构建时间序列输入
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])  # 输入形状: (time_steps, features)
        y.append(data[i + time_steps, -1])  # 目标为发电功率
    return np.array(X), np.array(y)




# 1D卷积特征提取
def build_cnn_feature_extractor(input_layer):
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    return pool1

# Transformer Encoder 层
def transformer_encoder(inputs, num_heads=4, ff_dim=128, dropout=0.3):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attention = LayerNormalization()(attention)
    return attention

# 残差块（Residual Block）
def residual_block(x):
    shortcut = x
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = Add()([x, shortcut])  # Residual connection
    return x

# ResNet-Transformer 风电功率预测模型
def build_resnet_transformer_model(input_shape):
    inputs = Input(shape=input_shape)

    # CNN特征提取
    cnn_features = build_cnn_feature_extractor(inputs)

    # Transformer建模长时依赖
    transformer_out = transformer_encoder(cnn_features)

    # 残差块帮助模型学习深层特征
    residual_out = residual_block(transformer_out)

    # LSTM部分：建模时间依赖
    lstm_out = LSTM(64, return_sequences=True)(residual_out)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(32)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)

    # 最后输出层
    outputs = Dense(1)(lstm_out)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    return model




# 4. 训练模型
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history


# 5. 预测与评估
def evaluate_model(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)
    # 提取第一个时间步的所有特征（包括目标变量）
    X_test_features = X_test[:, 0, :]  # 形状 (样本数, 12)

    # 反标准化预测值
    X_test_features_with_pred = X_test_features.copy()
    X_test_features_with_pred[:, -1] = y_pred.flatten()  # 替换最后一列为预测值
    y_pred_rescaled = scaler.inverse_transform(X_test_features_with_pred)[:, -1]

    # 反标准化真实值
    X_test_features_with_actual = X_test_features.copy()
    X_test_features_with_actual[:, -1] = y_test.flatten()
    y_test_rescaled = scaler.inverse_transform(X_test_features_with_actual)[:, -1]

    # 可视化对比
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    plt.figure(figsize=(80, 6))
    plt.plot(y_test_rescaled, label="真实值")
    plt.plot(y_pred_rescaled, label="预测值")
    plt.xlabel("时间步")
    plt.ylabel("发电功率")
    plt.legend()
    plt.title("CNN-LSTM预测结果对比")
    plt.show()

    # 计算指标
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    print(f"MAE: {mae}",f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")




# 主函数
def main():
    # 每次运行时重新初始化数据
    train_df, val_df, test_df, features, target, scaler = load_and_preprocess_data()
    # 2. 构建时间序列输入
    time_steps = 24
    train_data = np.hstack([train_df[features].values, train_df[[target]].values])
    X_train, y_train = create_sequences(train_data, time_steps)

    val_data = np.hstack([val_df[features].values, val_df[[target]].values])
    X_val, y_val = create_sequences(val_data, time_steps)

    test_data = np.hstack([test_df[features].values, test_df[[target]].values])
    X_test, y_test = create_sequences(test_data, time_steps)

    # 3. 构建CNN-LSTM模型
    input_shape = (time_steps, len(features)+1)
    model = build_resnet_transformer_model(input_shape)
    model.summary()

    # 4. 训练模型
    history = train_model(model, X_train, y_train, X_val, y_val)

    # 5. 预测与评估
    evaluate_model(model, X_test, y_test, scaler)


# 运行主函数
if __name__ == "__main__":
    main()