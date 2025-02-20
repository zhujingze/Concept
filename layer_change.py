import random
###shuffle
# 设置随机种子
seed = 42  # 可以设置为任意整数
random.seed(seed)

# 指定范围
start_idx = 5  # 起始索引（包含）
end_idx = 10   # 结束索引（包含）

# 提取范围内的层
layers_subset = model.model.layers[start_idx:end_idx + 1]

# 打乱顺序
random.shuffle(layers_subset)

# 打印打乱后的顺序
print("打乱后的层顺序（索引范围 {} 到 {}）：".format(start_idx, end_idx))
for i, layer in enumerate(layers_subset, start=start_idx):
    print("层索引 {} -> 原始层索引 {}".format(i, model.model.layers.index(layer)))

# 将打乱后的层重新赋值回原位置
for i in range(start_idx, end_idx + 1):
    model.model.layers[i] = layers_subset[i - start_idx]

###exchange
layer_idx_1 = 19  # 第20层
layer_idx_2 = 20  # 第21层

# 直接交换两个层的引用
model.model.layers[layer_idx_1], model.model.layers[layer_idx_2] = (
    model.model.layers[layer_idx_2],
    model.model.layers[layer_idx_1]
)

###replace
import copy

# 假设层索引从0开始（第20层对应索引19，第21层对应索引20）
src_layer_idx = 20  # 源层索引（第21层）
tgt_layer_idx = 19  # 目标层索引（第20层）

# 获取并复制源层
src_layer = model.model.layers[src_layer_idx]
cloned_layer = copy.deepcopy(src_layer)

# 替换目标层
model.model.layers[tgt_layer_idx] = cloned_layer
