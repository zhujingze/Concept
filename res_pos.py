def process_accuracy_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    group_num = 0  
    total_accuracy = 0  
    total_count = 0  
    current_group_end = -1  # 当前group的结束范围

    i = 0
    while i < len(lines):
        if 'Accuracy' in lines[i] and 'total num' in lines[i]:
            if i > current_group_end:  # 说明要开始新的group
                if total_count > 0:  # 打印上一个group的结果
                    print(f"Group: {group_num}, num: {total_count}, total: {total_accuracy}")
                    group_num += 1
                total_accuracy = 0  
                total_count = 0  

            # 计算新group的范围
            current_group_end = max(current_group_end, i + 5)

            try:
                accuracy_value = float(lines[i].split('Accuracy:')[1].split()[0])
                total_num_value = int(lines[i].split('total num')[1].split()[1])
                total_accuracy += accuracy_value * total_num_value
                total_count += total_num_value
            except IndexError:
                print(f"Skipping line {i} due to unexpected format: {lines[i].strip()}")

        i += 1  

    # 处理最后一个group
    if total_count > 0:
        print(f"Group: {group_num}, num: {total_count}, total: {total_accuracy}")

# 调用函数
file_path = 'your_file.txt'  
process_accuracy_file(file_path)
