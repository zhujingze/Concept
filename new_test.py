total_correct = []
total_scale = []
total_max = []
total_entropy = []  # 用于累积每层的熵总和
total_kl = []       # 用于累积每层的KL散度总和（层i对应与层i-1的KL）
total_samples = 0

def js_divergence(p, q, epsilon=1e-8):
    """
    计算两个概率分布之间的Jensen-Shannon散度。
    Args:
        p: 概率分布 (batch_size, num_classes)
        q: 概率分布 (batch_size, num_classes)
        epsilon: 防止log(0)的小值
    Returns:
        JS散度 (batch_size,)
    """
    m = 0.5 * (p + q)  # 中间分布
    kl_p_m = F.kl_div(torch.log(p + epsilon), m, reduction='none').sum(dim=-1)  # KL(P || M)
    kl_q_m = F.kl_div(torch.log(q + epsilon), m, reduction='none').sum(dim=-1)  # KL(Q || M)
    js = 0.5 * (kl_p_m + kl_q_m)  # JS(P || Q)
    return js

if args.method == "letter":
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)

        out_idxs = []
        for i in range(attention_mask.size(0)):
            out_idx = ((attention_mask[i] != 1).nonzero(as_tuple=True)[0])[0].item() - 1
            out_idxs.append(out_idx)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
        out_idxs = torch.tensor(out_idxs, device=device)
        out_idxs = out_idxs.unsqueeze(1)

        batch_size = label.size(0)
        prev_probs = None  # 保存前一层的概率分布

        for layer_idx in range(outputs.logits.size(0)):
            logits = outputs.logits[layer_idx]
            logits = logits.unsqueeze(0)
            logits = logits.gather(1, out_idxs.unsqueeze(-1).expand(-1, -1, logits.size(-1)).long())
            logits = logits.squeeze(1)

            # 提取四个特定token的logits并转换为概率分布
            logits = logits[:, [319, 350, 315, 360]]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # 计算熵 (H = -Σ p * log p)
            epsilon = 1e-8  # 防止log(0)
            entropy = - (probs * torch.log(probs + epsilon)).sum(dim=1)
            entropy_sum = entropy.sum().item()  # 当前层的熵总和

            # 更新total_entropy
            if len(total_entropy) <= layer_idx:
                total_entropy.append(entropy_sum)
            else:
                total_entropy[layer_idx] += entropy_sum

            # 计算KL散度（相对于前一层）
            if prev_probs is not None:
                # KL(p_current || p_prev)
                kl = (probs * (torch.log(probs + epsilon) - probs * torch.log(prev_probs + epsilon))).sum(dim=1)
                kl_sum = kl.sum().item()  # 当前层的KL总和

                # 确定KL对应的索引（layer_idx-1）
                kl_layer_idx = layer_idx - 1
                if len(total_kl) <= kl_layer_idx:
                    total_kl.append(kl_sum)
                else:
                    total_kl[kl_layer_idx] += kl_sum

            # 保存当前层概率供下一层使用
            prev_probs = probs.detach()

            # 原有准确率计算逻辑
            max_val, idx = torch.max(logits.flatten(), dim=-1)
            logits = logits[:, [319, 350, 315, 360]]
            max_val = torch.max(logits)
            scale = torch.floor(torch.log10(max_val)).item()
            max_val = max_val.item()

            if len(total_scale) < (layer_idx + 1):
                total_scale.append(scale)
                total_max.append(max_val)
            else:
                total_scale[layer_idx] += scale
                total_max[layer_idx] += max_val

            logits = logits.to(device)
            batch_accuracy = compute_accuracy(logits, label)
            if len(total_correct) < (layer_idx + 1):
                total_correct.append(batch_accuracy * batch_size)
            else:
                total_correct[layer_idx] += batch_accuracy * batch_size

        total_samples += batch_size

# 打印准确率
for idx, acc in enumerate(total_correct):
    print(f"Layer {idx+1} Acc: {acc / total_samples * 100:.2f}%")

# 打印每层的平均熵
for idx, entropy_sum in enumerate(total_entropy):
    avg_entropy = entropy_sum / total_samples
    print(f"Layer {idx+1} Entropy: {avg_entropy:.4f}")

# 打印每层的平均KL散度（层i对应与层i-1）
for kl_idx, kl_sum in enumerate(total_kl):
    avg_kl = kl_sum / total_samples
    layer_idx = kl_idx + 1  # KL散度对应层i与层i-1（i从1开始）
    print(f"Layer {layer_idx+1} KL Divergence (vs Layer {layer_idx}): {avg_kl:.4f}")

###
    def logits_to_onehot(logits):
    """
    将logits转换为one-hot形式。
    Args:
        logits: (batch_size, num_classes)
    Returns:
        onehot: (batch_size, num_classes)
    """
    max_indices = torch.argmax(logits, dim=-1)  # 找到最大值的位置
    onehot = torch.zeros_like(logits)
    onehot.scatter_(1, max_indices.unsqueeze(1), 1)  # 将最大值位置设为1
    return onehot
        # 收集各层one-hot结果（从第16层开始）
        layers_onehot = []
        for layer_idx in range(outputs.logits.size(0)):
            logits = outputs.logits[layer_idx].unsqueeze(0)
            logits = logits.gather(1, out_idxs.unsqueeze(-1).expand(-1, -1, logits.size(-1)).squeeze(1)
            logits = logits[:, [319, 350, 315, 360]]  # 提取特定token

            # 转换为one-hot形式
            onehot = logits_to_onehot(logits)
            if layer_idx >= 15:  # 第16层对应索引15（假设层索引从0开始）
                layers_onehot.append(onehot.detach())  # 避免梯度计算

            # 原有熵和准确率计算（保持不变）
            # ... [原有代码] ...

        # 计算一致性矩阵（仅当存在至少一个层时）
        if layers_onehot:
            layers_onehot = torch.stack(layers_onehot, dim=0)  # (num_layers, batch_size, 4)
            num_layers = layers_onehot.size(0)

            # 初始化一致性矩阵
            consistency_matrix = torch.zeros((num_layers, num_layers), device=device)

            # 计算所有层对之间的一致性
            for i in range(num_layers):
                for j in range(num_layers):
                    # 计算一致性（相同位置为1的比例）
                    agreement = (layers_onehot[i] == layers_onehot[j]).all(dim=-1).float().mean()
                    consistency_matrix[i, j] = agreement

            # 累积到总矩阵
            if total_consistency_matrix is None:
                total_consistency_matrix = consistency_matrix.cpu()
            else:
                total_consistency_matrix += consistency_matrix.cpu()
            total_samples += batch_size

        # 原有的准确率、熵等计算
        # ... [原有代码] ...

# 计算平均一致性矩阵并生成热力图
if total_consistency_matrix is not None and total_samples > 0:
    avg_consistency_matrix = total_consistency_matrix / total_samples
    num_layers = avg_consistency_matrix.size(0)
    layer_labels = [f"Layer {16 + i}" for i in range(num_layers)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        avg_consistency_matrix.numpy(),
        annot=True,
        fmt=".4f",
        xticklabels=layer_labels,
        yticklabels=layer_labels,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Consistency'}
    )
    plt.title("Decision Consistency Between Layers (Starting from Layer 16)")
    plt.xlabel("Target Layer (j)")
    plt.ylabel("Source Layer (i)")
    plt.tight_layout()
    plt.savefig("consistency_heatmap_layer16_onwards.png")
    plt.close()
else:
    print("No layers beyond 16 were processed for consistency calculation.")
###
