import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaWithLayerWeights
from dataset_mmlu import MultipleChoiceDataset
from torch.utils.data import DataLoader
import os

def compute_loss(logits, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss

device = args.device
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = LlamaWithLayerWeights.from_pretrained(args.model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

test_filename = args.subject + '_test.csv'
val_filename = args.subject + '_val.csv'

test_file = os.path.join(os.path.join(args.data_folder, 'test'), test_filename)
val_file = os.path.join(os.path.join(args.data_folder, 'val'), val_filename)

dataset = MultipleChoiceDataset(test_file, val_file, tokenizer)
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
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)

        out_idxs = []
        for i in range(args.bs):
            out_idx = ((attention_mask[i] != 1).nonzero(as_tuple=True)[0])[0].item() - 1
            out_idxs.append(out_idx)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
        out_idxs = torch.tensor(out_idxs, device = device)
        out_idxs = out_idxs.unsqueeze(1)
          
        logits = outputs.logits.gather(1, out_idxs.unsqueeze(-1).expand(-1, -1, outputs.logits.size(-1).long()))
        logits = logits。squeeze(1)
        logits = logits[:, [319, 350, 315, 360]]
        logits = logits.to(device)
        loss = compute_loss(logits, label)
        loss.backward()

        print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")

        optimizer.step()
        
    if args.save_folder:
        torch.save(model.layer_weights.data, os.path.join(args.save_floder, f"{args.subject}_epoch{epoch+1}.pth"))
    print(f"End of Epoch {epoch+1}, Layer Weights:", model.layer_weights.data)
