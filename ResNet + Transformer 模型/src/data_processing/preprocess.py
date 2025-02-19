import pandas as pd
import os

# 定义 Excel 文件路径
folder_path = '/Users/xutong/Documents/风电功率预测/ResNet + Transformer 模型/data/raw'  # 替换为你的文件夹路径
output_file = '/Users/xutong/Documents/风电功率预测/ResNet + Transformer 模型/data/user_data/汇总.xlsx'  # 输出合并后的文件路径

# 获取文件夹中所有 Excel 文件
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# 用来存储读取的 DataFrame
dfs = []

# 循环读取每个 Excel 文件
for file in excel_files:
    file_path = os.path.join(folder_path, file)
    # 读取 Excel 文件到 DataFrame
    df = pd.read_excel(file_path, sheet_name="Sheet1")  # 假设数据在“Sheet1”中
    dfs.append(df)

# 合并所有 DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# 将合并后的数据写入新的 Excel 文件
merged_df.to_excel(output_file, index=False)

print(f"合并完成！文件保存在: {output_file}")
