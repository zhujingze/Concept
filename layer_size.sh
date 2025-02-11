#!/bin/bash

# 循环 num 从 19 到 31
for num in {19..31}; do
    # 创建输出目录（如果不存在）
    mkdir -p "data/math/${num}"
    
    # 定义输出文件路径
    output_file="data/math/${num}/math.txt"
    
    # 执行命令并重定向输出
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 ddp.py \
        --total_epochs 10 \
        --save_every 1 \
        --batch_size 1 \
        --model 'Llama-2-7b' \
        --data_folder 'data' \
        --subject 'math' \
        --lr 1e-3 \
        --save_folder 'ckp' \
        --method "letter" \
        --num "${num}" > "${output_file}" 2>&1
    
    # 可选：添加间隔（例如等待10秒）
    sleep 10
done

# 定义一个包含多个 subject 的数组
subjects=("math" "physics" "chemistry" "biology" "history")

# 固定 num 的值
num=25

# 循环遍历 subjects 数组
for subject in "${subjects[@]}"; do
    # 创建输出目录（如果不存在）
    mkdir -p "data/${subject}/${num}"
    
    # 定义输出文件路径
    output_file="data/${subject}/${num}/${subject}.txt"
    
    # 执行命令并重定向输出
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 ddp.py \
        --total_epochs 10 \
        --save_every 1 \
        --batch_size 1 \
        --model 'Llama-2-7b' \
        --data_folder 'data' \
        --subject "${subject}" \
        --lr 1e-3 \
        --save_folder 'ckp' \
        --method "letter" \
        --num "${num}" > "${output_file}" 2>&1
    
    # 可选：添加间隔（例如等待10秒）
    sleep 10
done
