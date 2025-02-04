import torch
import re

def calculate_accuracy(file_path):
    correct = 0
    total = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        # 读取logits和label的行
        logits_line = lines[i].strip()
        label_line = lines[i+1].strip()

        # 如果是'Original'就结束
        if "Original" in logits_line:
            break

        # 提取logits中的数值
        logits_values = re.findall(r"[-+]?\d*\.\d+|\d+", logits_line)  # 提取数字
        logits = torch.tensor([float(x) for x in logits_values])

        # 提取label中的数值
        label_values = re.findall(r"[-+]?\d*\.\d+|\d+", label_line)  # 提取数字
        label = torch.tensor([int(x) for x in label_values])

        # 计算最大概率的索引
        predicted_label = torch.argmax(logits).item()

        # 判断是否正确
        if predicted_label == label.item():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# 使用示例
file_path = 'your_log_file.txt'  # 替换为你的文件路径
accuracy = calculate_accuracy(file_path)
print(f"Accuracy: {accuracy * 100:.2f}%")
