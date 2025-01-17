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

def compute_loss(logits, labels):
    criterion = nn.NLLLoss()
    loss = criterion(logits, labels)
    return loss

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
    if args.method == 'concat':
        dataset = MultipleChoiceConcatDataset(args.subject, test_file, val_file, tokenizer)
    if args.method == 'wo_option':
        dataset = MultipleChoiceConcatWODataset(args.subject, test_file, val_file, tokenizer)
        
    train_loader = DataLoader(dataset, batch_size = args.bs, shuffle = True)

    for param in model.parameters():
        param.requires_grad = False

    model.layer_weights.requires_grad = True
    
    from torch.optim import AdamW
    optimizer = AdamW([model.layer_weights], lr=args.lr)
    
    if args.save_folder:
        os.makedirs(save_folder, exist_ok=True)

    for epoch in range(args.epoch):
        model.train()
        if args.method == 'letter':
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
                loss = compute_loss(logits, label)
                loss.backward()
        
                print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")
        
                optimizer.step()
                
            if args.save_folder:
                torch.save(model.layer_weights.data, os.path.join(args.save_floder, f"{args.subject}_epoch{epoch+1}.pth"))
            print(f"End of Epoch {epoch+1}, Layer Weights:", model.layer_weights.data)
            # model.layer_weights.data = torch.clamp(model.layer_weights.data, min=0.0, max=1.0)
            # print(f"End of Epoch {epoch+1}, Layer Weights:", model.layer_weights.data / model.layer_weights.data.sun())
        if args.method == 'concat' or 'concat_wo_option':
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                prefix_ids_len = batch["prefix_ids_len"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)

                # batch
                all_batch = []
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
                    for j in range(attention_mask_sample.size(0)):
                        num = len(real_answer_ids[j])
                        start_idx = prefix_ids_len[i].to(device)
                        idx_range = torch.arange(num).unsqueeze(0).expand(1, num).to(device)
                        start_idx_tensor = start_idx.clone().unsqueeze(0).expand(1, num) - 2
                        final_idx = start_idx_tensor + idx_range
                        logits_selected = outputs.logits[j][final_idx]
                        real_answer_ids[j] = real_answer_ids[j].unsqueeze(0)
                        logits_selected = logits_selected[0, torch.arange(num), real_answer_ids[j].squeeze(0)]
                        logits_selected = logits_selected.sum() / num
                        one_batch.append(logits_selected)
                    one_batch = torch.stack(one_batch)
                    all_batch.append(one_batch)
                all_batch = torch.stack(all_batch, dim=0)
                logits = all_batch.to(device)
                loss = compute_loss(logits, label)
                loss.backward()
        
                print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")
        
                optimizer.step()
                
            if args.save_folder:
                torch.save(model.layer_weights.data, os.path.join(args.save_floder, f"{args.subject}_epoch{epoch+1}.pth"))
            print(f"End of Epoch {epoch+1}, Layer Weights:", model.layer_weights.data)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--subject', type=str)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--relative_top', type=float)
    parser.add_argument('--method', type=str)
    
    args = parser.parse_args()
    main(args)
