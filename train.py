import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaWithLayerWeights
from dataset_mmlu import MultipleChoiceDataset
from torch.utils.data import DataLoader
import os

def main(args):
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = LlamaWithLayerWeights.from_pretrained(args.model)
  
    test_filename = args.subject + '_test.csv'
    vale_filename = args.subject + '_val.csv'
    
    test_file = os.path.join(os.path.join(data_folder, 'test'), test_filename)
    val_file = os.path.join(os.path.join(data_folder, 'val'), val_filename)
    
    dataset = MultipleChoiceDataset(test_file, val_file, tokenizer)
    train_loader = DataLoader(dataset, batch_size = args.bs, shuffle = True)

  for param in model.parameters():
      param.requires_grad = False

  model.layer_weights.requires_grad = True

  from torch.optim import AdamW
  optimizer = AdamW([model.layer_weights], lr=args.lr):

  def compute_loss(logits, labels):
      criterion = nn.CrossEntropyLoss()
      loss = criterion(logits, labels)
      return loss
    
  if args.save_folder:
      os.makedirs(save_dir, exist_ok=True)

  for epoch in range(args.epoch):
      model.train()
      for batch in train_dataloader:
          optimizer.zero_grad()
          input_ids = batch["input_ids"].to(device)
          attention_mask = batch["attention_mask"].to(device)
          label = batch["label"].to(device)

          outputs = model(input_ids=input_ids, attention_mask=attention_mask)
          logits = outputs.logits[:,-1,:]
          logits = logits[:, [319, 350, 315, 360]]
          logits = logits.to(device)
          loss = compute_loss(logits, label)
          loss.backward()

          print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")

          optimizer.step()
      if args.save_folder:
          torch.save(model.layer_weights.data, os.path.join(args.save_floder, f"{args.subject}_epoch{epoch+1}.pth"))
      print(f"End of Epoch {epoch+1}, Layer Weights:", model.layer_weights.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--subject', type=str)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--lr', type=float')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--save_folder', type=str)
    
    # 解析命令行参数
    args = parser.parse_args()
    main(args)
