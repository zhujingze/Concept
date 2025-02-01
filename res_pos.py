def process_accuracy_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    accuracy_lines = []
    for i, line in enumerate(lines):
        if 'Accuracy' in line:
            # 获取该行的Accuracy和total num值
            accuracy_value = float(line.split('Accuracy:')[1].split()[0])
            total_num_value = int(line.split('total num')[1].split()[1])
            accuracy_lines.append((i, accuracy_value, total_num_value))
    
    group_num = 0
    for idx, accuracy_value, total_num_value in accuracy_lines:
        # 定义每个Group的上下范围
        group_start = max(0, idx - 5)
        group_end = min(len(lines), idx + 6)
        
        total_accuracy = 0
        total_count = 0
        
        # 遍历该范围内的行
        for i in range(group_start, group_end):
            if 'Accuracy' in lines[i]:
                # 获取该行的Accuracy和total num值
                accuracy_value_in_group = float(lines[i].split('Accuracy:')[1].split()[0])
                total_num_value_in_group = int(lines[i].split('total num')[1].split()[1])
                total_accuracy += accuracy_value_in_group * total_num_value_in_group
                total_count += total_num_value_in_group
        
        print(f"Group: {group_num}, num: {len(range(group_start, group_end))}, total: {total_accuracy}")
        group_num += 1

# 调用函数
file_path = 'your_file.txt'  # 替换成你的文件路径
process_accuracy_file(file_path)
