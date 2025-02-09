import os
import re
import pandas as pd

base_dir = "option_size"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 初始化存储结构
accuracy_data = {}
average_data = {}

# 遍历所有子文件夹（0-31）
for folder in sorted(os.listdir(base_dir), key=int):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    accuracy_data[folder] = {}
    average_data[folder] = {}
    
    # 遍历每个txt文件
    for file in sorted(os.listdir(folder_path)):
        if not file.endswith(".txt"):
            continue
            
        file_path = os.path.join(folder_path, file)
        numbers = []
        accuracy = None
        
        # 解析文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 匹配准确率行（支持整数和小数）
                if line.startswith("Original Accuracy:"):
                    match = re.search(r"Original Accuracy:\s*([\d.]+)%", line)
                    if match:
                        accuracy = float(match.group(1))
                # 收集其他数字（排除空行和准确率行）
                elif line and not line.startswith("Original Accuracy"):
                    try:
                        numbers.append(float(line))
                    except ValueError:
                        continue
        
        # 存储结果
        accuracy_data[folder][file] = accuracy
        average_data[folder][file] = sum(numbers)/len(numbers) if numbers else 0.0

# 转换为DataFrame
def create_df(data_dict, columns_order=None):
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.index.name = "Folder"
    df.index = df.index.astype(int)  # 确保文件夹按数字排序
    df = df.sort_index()
    if columns_order:
        df = df[columns_order]  # 保持列顺序
    return df.round(2)  # 保留两位小数

# 获取所有文件名（假设所有文件夹文件相同）
sample_folder = os.listdir(os.path.join(base_dir, "0"))
txt_files = sorted([f for f in sample_folder if f.endswith(".txt")])

# 创建并保存准确率表格
accuracy_df = create_df(accuracy_data, txt_files)
accuracy_df.to_csv(os.path.join(output_dir, "accuracy_table.csv"))

# 创建并保存平均值表格
average_df = create_df(average_data, txt_files)
average_df.to_csv(os.path.join(output_dir, "average_table.csv"))

print("统计完成！结果已保存至 results 目录")
print("准确率表示例：")
print(accuracy_df.head())
print("\n平均值表示例：")
print(average_df.head())
