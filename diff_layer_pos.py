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
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                # 匹配准确率行
                if line.startswith("Original Accuracy:"):
                    match = re.search(r"(\d+)%", line)
                    if match:
                        accuracy = int(match.group(1))
                # 收集其他数字
                else:
                    try:
                        numbers.append(float(line))
                    except ValueError:
                        continue
        
        # 存储结果
        accuracy_data[folder][file] = accuracy
        average_data[folder][file] = sum(numbers)/len(numbers) if numbers else 0

# 转换为DataFrame
def create_df(data_dict, value_name):
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.index.name = "Folder"
    df.columns.name = value_name
    return df

# 创建并保存准确率表格
accuracy_df = create_df(accuracy_data, "Accuracy (%)")
accuracy_df.to_csv(os.path.join(output_dir, "accuracy_table.csv"))

# 创建并保存平均值表格
average_df = create_df(average_data, "Average Value")
average_df.to_csv(os.path.join(output_dir, "average_table.csv"))

print("统计完成！结果已保存至 results 目录")
