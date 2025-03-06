import pandas as pd 
from torch.utils.data import Dataset

class MovieDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_text = str(self.data.iloc[index]['input'])
        target_text = str(self.data.iloc[index]['response'])
        
        source = self.tokenizer(
            input_text, 
            padding='max_length', 
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            target_text, 
            padding='max_length', 
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        source_ids = source.input_ids.squeeze()
        target_ids = target.input_ids.squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source.attention_mask.squeeze(),
            "labels": target_ids
        }