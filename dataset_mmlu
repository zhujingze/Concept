from torch.utils.data import Dataset
import pandas as pd
import torch

class MultipleChoiceDataset(Dataset):
    def __init__(self, csv_file, example_file, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file, header=None)
        self.example_data = pd.read_csv(example_file, header=None)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.choices = ["A", "B", "C", "D"]

    def __len__(self):
        return len(self.data)

    def get_example(self):
        example_text = ''
        for idx in range(5):
            question = self.example_data.iloc[idx, 0]
            choices = [self.example_data.iloc[idx, i+1] for i in range(4)]
            answer = self.example_data.iloc[idx, 5]
            example_text += "<question>:\n"
            example_text += f"{question}\n"
            example_text += "<options>:\n"
            for i, choice in enumerate(choices):
                example_text += f"{self.choices[i]}. {choice}\n"
            example_text += "<answer>:\n"
            example_text += f"{answer}\n\n"
        return example_text

    def __getitem__(self, idx):
        question = self.data.iloc[idx, 0]
        choices = [self.data.iloc[idx, i+1] for i in range(4)]
        answer = self.data.iloc[idx, 5]
        prompt_text = "Choose the right answer from options the answer should be one of A, B, C, D.\n"
        prompt_text += self.get_example()

        input_text += "<question>:\n"
        input_text += f"{question}\n"
        input_text += "<options>:\n"
        for i, choice in enumerate(choices):
            input_text += f"{self.choices[i]}. {choice}\n"
        input_text += "<answer>: \n"
        input_text = prompt_text + input_text

        encodings = self.tokenizer(input_text, return_tensors='pt')
        label = self.choices.index(answer)
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "label": torch.tensor(label)
        }
