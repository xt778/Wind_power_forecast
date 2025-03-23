import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Add, LayerNormalization, MultiHeadAttention, \
    Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pylab import mpl

# 1. 数据加载与预处理
def load_and_preprocess_data():
    merged_file_path = os.path.join(
        'D:\\PycharmProjects\\Wind_power_forecast\\ResNet + Transformer 模型\\data\\user_data', '汇总.xlsx')
    df = pd.read_excel(merged_file_path, sheet_name="Sheet1")

    df["时间"] = pd.to_datetime(df["年份"])
    df.set_index("时间", inplace=True)
    df["小时"] = df.index.hour
    df["星期几"] = df.index.dayofweek
    df["月份"] = df.index.month

    # 添加滞后特征
    lag_steps = 4
    for lag in range(1, lag_steps + 1):
        df[f"风速_lag{lag}"] = df["风速"].shift(lag)
        df[f"发电功率_lag{lag}"] = df["发电功率"].shift(lag)
    df.dropna(inplace=True)

    # 标准化
    features = ["空气密度", "风速", "小时", "星期几", "月份"] + [col for col in df.columns if "lag" in col]
    target = "发电功率"

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # 分割数据集
    train_size = int(0.9 * len(df))
    val_size = int(0.08 * len(df))
    train_df, val_df, test_df = df.iloc[:train_size], df.iloc[train_size:train_size + val_size], df.iloc[
                                                                                                 train_size + val_size:]

    return train_df, val_df, test_df, features, target, scaler


# 2. 构建时间序列输入
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        y.append(data[i + time_steps, -1])
    return np.array(X), np.array(y)


# 3. ResNet 残差块
def residual_block(x, filters=64):
    shortcut = x
    x = Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Add()([x, shortcut])  # 残差连接
    x = LayerNormalization()(x)  # 归一化
    return x


# 4. Transformer Encoder 层
def transformer_encoder(inputs, num_heads=4, ff_dim=128, dropout=0.3):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attention = Add()([inputs, attention])  # 残差连接
    attention = LayerNormalization()(attention)

    ffn_output = Dense(ff_dim, activation="relu")(attention)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    return LayerNormalization()(Add()([attention, ffn_output]))


# 5. 贯穿 ResNet 的 ResNet-Transformer-LSTM 预测模型
def build_resnet_transformer_model(input_shape):
    inputs = Input(shape=input_shape)

    # 1D CNN 处理输入特征
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)

    # 贯穿多个残差块
    for _ in range(3):  # 堆叠多个 ResNet 层
        x = residual_block(x)

    # Transformer 处理全局依赖关系
    x = transformer_encoder(x)

    # 再次加入 ResNet 残差块
    for _ in range(2):
        x = residual_block(x)

    # LSTM 建模时间序列
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)

    # 输出层
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    return model


# 6. 训练模型
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history


# 7. 预测与评估
def evaluate_model(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)

    X_test_features = X_test[:, 0, :]
    X_test_features[:, -1] = y_pred.flatten()
    y_pred_rescaled = scaler.inverse_transform(X_test_features)[:, -1]

    X_test_features[:, -1] = y_test.flatten()
    y_test_rescaled = scaler.inverse_transform(X_test_features)[:, -1]


    mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label="真实值")
    plt.plot(y_pred_rescaled, label="预测值")
    plt.legend()
    plt.title("ResNet-Transformer-LSTM 预测对比")
    plt.show()

    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    print(f"MAE: {mae}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")


# 8. 主函数
def main():
    train_df, val_df, test_df, features, target, scaler = load_and_preprocess_data()
    time_steps = 24

    train_data = train_df[features].values
    val_data = val_df[features].values
    test_data = test_df[features].values

    X_train, y_train = create_sequences(train_data, time_steps)
    X_val, y_val = create_sequences(val_data, time_steps)
    X_test, y_test = create_sequences(test_data, time_steps)

    model = build_resnet_transformer_model((time_steps, len(features)))
    model.summary()
    train_model(model, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test, scaler)


if __name__ == "__main__":
    main()
