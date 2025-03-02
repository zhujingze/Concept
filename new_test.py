total_correct = []
total_scale = []
total_max = []
total_entropy = []  # 用于累积每层的熵总和
total_kl = []       # 用于累积每层的KL散度总和（层i对应与层i-1的KL）
total_samples = 0

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
