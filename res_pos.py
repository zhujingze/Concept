def process_accuracy_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    accuracy_lines = []  # 保存所有含Accuracy行的索引和相应的数值
    group_num = 0  # 组号
    visited_lines = set()  # 记录已访问过的行

    i = 0
    while i < len(lines):
        if 'Accuracy' in lines[i] and 'total num' in lines[i]:
            # 找到包含Accuracy的行，初始化一个group
            group_start = max(0, i - 5)  # 向上取5行
            group_end = min(len(lines), i + 6)  # 向下取5行

            total_accuracy = 0
            total_count = 0

            # 从当前Accuracy行向上下5行内找到所有Accuracy行进行计算
            while i < len(lines) and 'Accuracy' in lines[i] and 'total num' in lines[i]:
                try:
                    accuracy_value_in_group = float(lines[i].split('Accuracy:')[1].split()[0])
                    total_num_value_in_group = int(lines[i].split('total num')[1].split()[1])
                    total_accuracy += accuracy_value_in_group * total_num_value_in_group
                    total_count += total_num_value_in_group
                    visited_lines.add(i)  # 标记当前行已经处理过
                except IndexError:
                    print(f"Skipping line {i} due to unexpected format: {lines[i].strip()}")

                i += 1  # 移动到下一行

            print(f"Group: {group_num}, num: {total_count}, total: {total_accuracy}")
            group_num += 1
        else:
            i += 1  # 如果当前行不包含Accuracy，继续下一个行

# 调用函数
file_path = 'your_file.txt'  # 替换成你的文件路径
process_accuracy_file(file_path)
