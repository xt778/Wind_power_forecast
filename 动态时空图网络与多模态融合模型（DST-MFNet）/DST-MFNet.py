import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Multiply, Add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCNConv, GraphAttention
from spektral.utils import normalized_adjacency


# 1. 增强数据预处理（添加图结构构建）
def load_and_preprocess_data():
    # ... [保持原有数据加载逻辑不变] ...
    folder_path = '/Users/xutong/Documents/风电功率预测/ResNet + Transformer 模型/data/raw'
    merged_file_path = os.path.join('/Users/xutong/Documents/风电功率预测/ResNet + Transformer 模型/data/user_data','汇总.xlsx')
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
    # 添加虚拟空间坐标（示例）
    df["x_coord"] = np.random.rand(len(df))  # 模拟空间坐标
    df["y_coord"] = np.random.rand(len(df))
    return train_df, val_df, test_df, features, target, scaler
# 2. 构建时间序列输入
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])  # 输入形状: (time_steps, features)
        y.append(data[i + time_steps, -1])  # 目标为发电功率
    return np.array(X), np.array(y)


def build_dynamic_graph(sequences):
    """动态构建图邻接矩阵（基于特征相似性）"""
    batch_size, time_steps, n_features = sequences.shape
    adj_matrices = []
    for b in range(batch_size):
        # 计算余弦相似度作为邻接矩阵
        sim_matrix = np.zeros((time_steps, time_steps))
        for i in range(time_steps):
            for j in range(time_steps):
                sim = np.dot(sequences[b, i, :], sequences[b, j, :]) / (
                        np.linalg.norm(sequences[b, i, :]) * np.linalg.norm(sequences[b, j, :]) + 1e-8)
                sim_matrix[i, j] = sim
        adj_matrices.append(normalized_adjacency(sim_matrix))
    return np.array(adj_matrices)


# 2. 构建DST-MFNet模型
class DynamicSpatialTemporal(Model):
    def __init__(self, time_steps, n_features):
        super().__init__()
        # 多模态分支
        self.temporal_branch = LSTM(64, return_sequences=True)
        self.spatial_branch = GCNConv(64, activation='relu')

        # 动态图注意力
        self.gat = GraphAttention(64, attn_heads=4)

        # 特征融合
        self.fusion_dense = Dense(128, activation='relu')
        self.output_layer = Dense(1)

    def call(self, inputs):
        # 输入形状: (batch_size, time_steps, n_features)
        x, adj = inputs

        # 时空特征提取
        temporal_feat = self.temporal_branch(x)  # (batch, time, 64)

        # 空间特征提取（应用图卷积）
        spatial_feat = []
        for t in range(x.shape[1]):
            node_feat = x[:, t, :]  # (batch, n_features)
            spatial_feat_t = self.spatial_branch([node_feat, adj])  # (batch, 64)
            spatial_feat.append(spatial_feat_t)
        spatial_feat = tf.stack(spatial_feat, axis=1)  # (batch, time, 64)

        # 动态图注意力
        gat_out = self.gat([spatial_feat, adj])

        # 多模态融合
        fused = Add()([temporal_feat, gat_out])
        fused = LayerNormalization()(fused)
        fused = self.fusion_dense(fused)

        # 输出预测
        return self.output_layer(fused[:, -1, :])


# 3. 修改模型构建函数
def build_dst_mfnet(time_steps, n_features):
    x_input = Input(shape=(time_steps, n_features))
    adj_input = Input(shape=(time_steps, time_steps))

    model = DynamicSpatialTemporal(time_steps, n_features)
    predictions = model([x_input, adj_input])

    full_model = Model(inputs=[x_input, adj_input], outputs=predictions)
    full_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return full_model


# 4. 修改训练流程
def train_model(model, X_train, X_val, y_train, y_val):
    # 动态生成图结构
    train_adj = build_dynamic_graph(X_train)
    val_adj = build_dynamic_graph(X_val)

    history = model.fit(
        [X_train, train_adj], y_train,
        validation_data=([X_val, val_adj], y_val),
        epochs=50, batch_size=64, verbose=1
    )
    return history


# 5. 修改评估流程
def evaluate_model(model, X_test, y_test, scaler):
    test_adj = build_dynamic_graph(X_test)
    y_pred = model.predict([X_test, test_adj])

    # ... [保持原有反标准化和评估逻辑不变] ...



# 主函数调整
def main():
    train_df, val_df, test_df, features, target, scaler = load_and_preprocess_data()

    time_steps = 24
    # 构建序列数据（保持原有逻辑）
    train_data = np.hstack([train_df[features].values, train_df[[target]].values])
    X_train, y_train = create_sequences(train_data, time_steps)
    val_data = np.hstack([val_df[features].values, val_df[[target]].values])
    X_val, y_val = create_sequences(val_data, time_steps)

    test_data = np.hstack([test_df[features].values, test_df[[target]].values])
    X_test, y_test = create_sequences(test_data, time_steps)

    # ... [验证集和测试集处理相同] ...

    # 获取特征维度
    n_features = X_train.shape[-1]

    # 构建模型
    model = build_dst_mfnet(time_steps, n_features)
    model.summary()

    # 训练与评估
    history = train_model(model, X_train, X_val, y_train, y_val)
    evaluate_model(model, X_test, y_test, scaler)


if __name__ == "__main__":
    main()