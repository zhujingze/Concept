def extract_lines(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()  # 读取所有行
        
    extracted_lines = []  # 存储提取的行数据
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. 查找包含“Layer Weights”的行及接下来的6行（共7行）
        if "Layer Weights" in line:
            extracted_lines.append(line.strip())  # 添加当前行
            # 添加接下来的6行
            for j in range(1, 7):
                if i + j < len(lines):  # 防止索引越界
                    extracted_lines.append(lines[i + j].strip())
            i += 7  # 跳过接下来的6行
            continue  # 跳过后续的处理
        
        # 2. 查找包含“Accuracy”的行
        if "Accuracy" in line:
            extracted_lines.append(line.strip())
        
        i += 1  # 处理下一行

    # 删除包含"prompt"的行
    extracted_lines = [line for line in extracted_lines if "prompt" not in line.lower()]
    
    # 将提取的内容保存到输出文件
    with open(output_file, 'w') as output:
        for line in extracted_lines:
            output.write(line + '\n')

    print(f"Extracted lines have been saved to {output_file}")

# 调用函数进行提取
input_file = 'log.txt'  # 输入的txt文件
output_file = 'extracted.txt'  # 输出的txt文件
extract_lines(input_file, output_file)
