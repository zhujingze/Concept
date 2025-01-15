from torch.utils.data import Dataset
import pandas as pd
import torch

class MultipleChoiceDataset(Dataset):
    def __init__(self, subject, csv_file, example_file, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file, header=None)
        self.example_data = pd.read_csv(example_file, header=None)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.choices = ["A", "B", "C", "D"]
        self.subject = subject

    def __len__(self):
        return len(self.data)

    def get_example(self):
        example_text = ''
        for idx in range(5):
            question = self.example_data.iloc[idx, 0]
            choices = [self.example_data.iloc[idx, i+1] for i in range(4)]
            answer = self.example_data.iloc[idx, 5]
            #example_text += "<question>:\n"
            example_text += f"{question}\n"
            #example_text += "<options>:\n"
            for i, choice in enumerate(choices):
                example_text += f"{self.choices[i]}. {choice}\n"

            correct_choice_idx = self.choices.index(answer)
            correct_answer_text = choices[correct_choice_idx]
            example_text += f"Answer: {self.choices[correct_choice_idx]}. {correct_answer_text}\n\n"

        return example_text

    def __getitem__(self, idx):
        question = self.data.iloc[idx, 0]
        choices = [self.data.iloc[idx, i+1] for i in range(4)]
        answer = self.data.iloc[idx, 5]
        prompt_text = f"The following are multiple choice questions (with answers) about {self.subject}.\n\n"
        prompt_text += self.get_example()
        answer_text = []
        final_text = []
        #input_text = "<question>:\n"
        input_text = f"{question}\n"
        #input_text += "<options>:\n"
        for i, choice in enumerate(choices):
            input_text += f"{self.choices[i]}. {choice}\n"
            answer_text.append(f"{self.choices[i]}. {choice}\n")
        input_text += "Answer: "
        input_text = prompt_text + input_text
        prefix = self.tokenizer(input_text, return_tensors='pt')
        
        for i in range(len(choices)):
            final_text.append(input_text + answer_text[i])
            
        encodings = self.tokenizer(
            final_text,
            padding = "max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        label = self.choices.index(answer)
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "label": torch.tensor(label)
        }
