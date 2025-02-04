import re

# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

# 判断label是否与logits中的最大值对应
def check_correctness(logits_line, label_line):
    # 确保logits_line不为空且包含预期的格式，支持科学计数法
    logits_match = re.findall(r'logits\s*tensor\(\[([0-9.e+\-,\s]+)\]', logits_line)
    if not logits_match:
        return False  # 如果没有找到匹配的logits，返回错误

    logits = logits_match[0]
    logits = list(map(float, logits.split(',')))  # 转换为浮动列表

    # 从label_line中提取label值
    label_match = re.findall(r'label\s*tensor\(\[([0-9]+)\]', label_line)
    if not label_match:
        return False  # 如果没有找到匹配的label，返回错误
    
    label = int(label_match[0])

    # 获取logits中最大值的索引
    predicted_label = logits.index(max(logits))

    # 判断label是否与最大值的索引对应
    return predicted_label == label

# 统计正确率
def calculate_accuracy(file_path):
    lines = read_file(file_path)
    total = 0
    correct = 0
    for i in range(0, len(lines), 2):
        if 'Original' in lines[i]:
            break
        logits_line = lines[i]
        label_line = lines[i + 1]
        
        if check_correctness(logits_line, label_line):
            correct += 1
        total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy

# 使用方法
file_path = 'your_file.txt'  # 修改为实际文件路径
accuracy = calculate_accuracy(file_path)
print(f"Correctness rate: {accuracy:.2f}%")
