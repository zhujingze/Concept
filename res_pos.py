def process_accuracy_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 用来保存所有包含Accuracy的行的索引和对应的数值
    accuracy_lines = []

    # 遍历所有行，找出包含Accuracy的行
    for i, line in enumerate(lines):
        if 'Accuracy' in line and 'total num' in line:
            try:
                accuracy_value = float(line.split('Accuracy:')[1].split()[0])
                total_num_value = int(line.split('total num')[1].split()[1])
                accuracy_lines.append((i, accuracy_value, total_num_value))
            except IndexError:
                print(f"Skipping line {i} due to unexpected format: {line.strip()}")

    group_num = 0
    visited_lines = set()  # 记录已经被归为某个组的行

    for idx, accuracy_value, total_num_value in accuracy_lines:
        if idx in visited_lines:
            continue  # 如果当前行已经被归为某个组，跳过

        group_start = max(0, idx - 5)  # 向上取5行
        group_end = min(len(lines), idx + 6)  # 向下取5行

        total_accuracy = 0
        total_count = 0

        # 遍历该组的所有行
        for i in range(group_start, group_end):
            if 'Accuracy' in lines[i] and 'total num' in lines[i]:
                try:
                    accuracy_value_in_group = float(lines[i].split('Accuracy:')[1].split()[0])
                    total_num_value_in_group = int(lines[i].split('total num')[1].split()[1])
                    total_accuracy += accuracy_value_in_group * total_num_value_in_group
                    total_count += total_num_value_in_group
                    visited_lines.add(i)  # 记录该行已经被处理
                except IndexError:
                    print(f"Skipping line {i} due to unexpected format: {lines[i].strip()}")
        
        print(f"Group: {group_num}, num: {total_count}, total: {total_accuracy}")
        group_num += 1

# 调用函数
file_path = 'your_file.txt'  # 替换成你的文件路径
process_accuracy_file(file_path)
