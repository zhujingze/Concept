from torch.utils.data import Dataset
import pandas as pd
import torch

class MultipleChoiceConcatDataset(Dataset):
    def __init__(self, subject, csv_file, example_file, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file, header=None)
        self.example_data = pd.read_csv(example_file, header=None)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subject = subject

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx, 0]
        choices = [self.data.iloc[idx, i+1] for i in range(4)]
        answer = self.data.iloc[idx, 5]
        prompt_text = f"The following are multiple choice questions (with answers) about {self.subject}.\n\n"
        answer_text = []
        final_text = []
        #input_text = "<question>:\n"
        input_text = f"Question: {question}\n"
        #input_text += "<options>:\n"
        for choice in choices:
            answer_text.append(f"{choice}")
        input_text += "Answer: "
        input_text = prompt_text + input_text
        
        for i in range(len(choices)):
            final_text.append(input_text + answer_text[i])

        prefix_encoding = self.tokenizer(input_text, return_tensors='pt')
        encodings = self.tokenizer(
            final_text,
            padding = "max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        label = self.choices.index(answer)
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "prefix_ids_len": len(prefix_encoding["input_ids"][0]),
            "label": torch.tensor(label)
        }