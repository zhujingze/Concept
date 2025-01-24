import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaWithLayerWeights
from dataset_mmlu import MultipleChoiceDataset
from dataset_mmlu_concat import MultipleChoiceConcatDataset
from dataset_mmlu_wo_option import MultipleChoiceConcatWODataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def compute_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def main(args):
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = LlamaWithLayerWeights.from_pretrained(args.model).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
  
    test_filename = args.subject + '_test.csv'
    val_filename = args.subject + '_val.csv'
    
    test_file = os.path.join(os.path.join(args.data_folder, 'test'), test_filename)
    val_file = os.path.join(os.path.join(args.data_folder, 'val'), val_filename)

    if args.method == 'letter':
        dataset = MultipleChoiceDataset(args.subject, test_file, val_file, tokenizer)
    if args.method == 'wo_option':
        dataset = MultipleChoiceConcatWODataset(args.subject, test_file, val_file, tokenizer)
        
    train_loader = DataLoader(dataset, batch_size = args.bs, shuffle = True)
    
    if args.save_folder:
        os.makedirs(save_folder, exist_ok=True)

    #ori
    total_correct = 0
    total_correct_norm = 0
    total_samples = 0
    if args.method == "letter":
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            out_idxs = []
            for i in range(attention_mask.size(0)):
                out_idx = ((attention_mask[i] != 1).nonzero(as_tuple=True)[0])[0].item() - 1
                out_idxs.append(out_idx)
    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
            out_idxs = torch.tensor(out_idxs, device = device)
            out_idxs = out_idxs.unsqueeze(1)
              
            logits = outputs.logits.gather(1, out_idxs.unsqueeze(-1).expand(-1, -1, outputs.logits.size(-1)).long())
            logits = logits.squeeze(1)
            logits = logits[:, [319, 350, 315, 360]]
            logits = logits.to(device)
            batch_accuracy = compute_accuracy(logits, label)
            total_correct += (batch_accuracy * label.size(0))
            total_samples += label.size(0)

    if args.method in ['concat', 'wo_option']:
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            prefix_ids_len = batch["prefix_ids_len"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            # batch
            all_batch = []
            all_batch_norm = []
            for i in range(attention_mask.size(0)):
                input_ids_sample = input_ids[i]
                attention_mask_sample = attention_mask[i]
                prefix_len = prefix_ids_len[i]
                answer_ids = input_ids_sample[:, prefix_len-1:]
                real_answer_ids = []
                for row in answer_ids:
                    idx = (row != 2).nonzero(as_tuple=True)[0].max().item()
                    row = row[:idx+1]
                    real_answer_ids.append(row)

                outputs = model(input_ids=input_ids_sample, attention_mask=attention_mask_sample)

                # option
                one_batch = []
                one_batch_norm = []
                for j in range(attention_mask_sample.size(0)):
                    num = len(real_answer_ids[j])
                    start_idx = prefix_ids_len[i].to(device)
                    idx_range = torch.arange(num).unsqueeze(0).expand(1, num).to(device)
                    start_idx_tensor = start_idx.clone().unsqueeze(0).expand(1, num) - 2
                    final_idx = start_idx_tensor + idx_range
                    
                    logits_selected = outputs.logits[j][final_idx]
                    real_answer_ids[j] = real_answer_ids[j].unsqueeze(0)
                    logits_selected = logits_selected[0, torch.arange(num), real_answer_ids[j].squeeze(0)]
                    logits_selected = logits_selected.sum()
                    logits_selected_norm = logits_selected.sum() / num
                    
                    one_batch.append(logits_selected)
                    one_batch_norm.append(logits_selected_norm)
                one_batch = torch.stack(one_batch)
                one_batch_norm = torch.stack(one_batch_norm)
                all_batch.append(one_batch)
                all_batch_norm.append(one_batch_norm)
            all_batch = torch.stack(all_batch, dim=0)
            all_batch_norm = torch.stack(all_batch_norm, dim=0)
            logits = all_batch.to(device)
            logits_norm = all_batch_norm.to(device)

            batch_accuracy = compute_accuracy(logits, label)
            total_correct += (batch_accuracy * label.size(0))
            total_samples += label.size(0)

            batch_accuracy_norm = compute_accuracy(logits_norm, label)
            total_correct_norm += (batch_accuracy_norm * label.size(0))

    epoch_accuracy = total_correct / total_samples
    print(f"Original Accuracy: {epoch_accuracy * 100:.2f}%")

    if args.method in ['concat', 'wo_option']:
        epoch_accuracy_norm = total_correct_norm / total_samples
        print(f"Original Accuracy_Norm: {epoch_accuracy_norm * 100:.2f}%")
    
    for epoch in range(args.epoch):
        #trained res
        saved_weights = torch.load(os.path.join(args.save_folder, f"{args.subject}_epoch{epoch}.pth"))
        model.layer_weights.data = saved_weights  # 直接将加载的权重赋值给对应的层
        
        total_correct = 0
        total_correct_norm = 0
        total_samples = 0
        if args.method == "letter":
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
    
                out_idxs = []
                for i in range(attention_mask.size(0)):
                    out_idx = ((attention_mask[i] != 1).nonzero(as_tuple=True)[0])[0].item() - 1
                    out_idxs.append(out_idx)
        
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
                out_idxs = torch.tensor(out_idxs, device = device)
                out_idxs = out_idxs.unsqueeze(1)
                  
                logits = outputs.logits.gather(1, out_idxs.unsqueeze(-1).expand(-1, -1, outputs.logits.size(-1)).long())
                logits = logits.squeeze(1)
                logits = logits[:, [319, 350, 315, 360]]
                logits = logits.to(device)
                batch_accuracy = compute_accuracy(logits, label)
                total_correct += (batch_accuracy * label.size(0))
                total_samples += label.size(0)
    
        if args.method in ['concat', 'wo_option']:
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                prefix_ids_len = batch["prefix_ids_len"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
    
                # batch
                all_batch = []
                all_batch_norm = []
                for i in range(attention_mask.size(0)):
                    input_ids_sample = input_ids[i]
                    attention_mask_sample = attention_mask[i]
                    prefix_len = prefix_ids_len[i]
                    answer_ids = input_ids_sample[:, prefix_len-1:]
                    real_answer_ids = []
                    for row in answer_ids:
                        idx = (row != 2).nonzero(as_tuple=True)[0].max().item()
                        row = row[:idx+1]
                        real_answer_ids.append(row)
    
                    outputs = model(input_ids=input_ids_sample, attention_mask=attention_mask_sample)
    
                    # option
                    one_batch = []
                    one_batch_norm = []
                    for j in range(attention_mask_sample.size(0)):
                        num = len(real_answer_ids[j])
                        start_idx = prefix_ids_len[i].to(device)
                        idx_range = torch.arange(num).unsqueeze(0).expand(1, num).to(device)
                        start_idx_tensor = start_idx.clone().unsqueeze(0).expand(1, num) - 2
                        final_idx = start_idx_tensor + idx_range
                        
                        logits_selected = outputs.logits[j][final_idx]
                        real_answer_ids[j] = real_answer_ids[j].unsqueeze(0)
                        logits_selected = logits_selected[0, torch.arange(num), real_answer_ids[j].squeeze(0)]
                        logits_selected = logits_selected.sum()
                        logits_selected_norm = logits_selected.sum() / num
                        
                        one_batch.append(logits_selected)
                        one_batch_norm.append(logits_selected_norm)
                    one_batch = torch.stack(one_batch)
                    one_batch_norm = torch.stack(one_batch_norm)
                    all_batch.append(one_batch)
                    all_batch_norm.append(one_batch_norm)
                all_batch = torch.stack(all_batch, dim=0)
                all_batch_norm = torch.stack(all_batch_norm, dim=0)
                logits = all_batch.to(device)
                logits_norm = all_batch_norm.to(device)
    
                batch_accuracy = compute_accuracy(logits, label)
                total_correct += (batch_accuracy * label.size(0))
                total_samples += label.size(0)
    
                batch_accuracy_norm = compute_accuracy(logits_norm, label)
                total_correct_norm += (batch_accuracy_norm * label.size(0))
    
        epoch_accuracy = total_correct / total_samples
        print(f"Epoch{epoch} Accuracy: {epoch_accuracy * 100:.2f}%")
    
        if args.method in ['concat', 'wo_option']:
            epoch_accuracy_norm = total_correct_norm / total_samples
            print(f"Epoch{epoch} Accuracy_Norm: {epoch_accuracy_norm * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--subject', type=str)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--method', type=str)
    
    args = parser.parse_args()
    main(args)
