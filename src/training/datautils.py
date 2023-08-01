from datasets import Dataset

# Define a custom dataset for finetuning
class AutoCorrectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self._data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inputs, target = self._data[idx]
        model_inputs = self.tokenizer(inputs, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        labels = self.tokenizer(text_target=target, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        model_inputs['labels'] = labels['input_ids']

        return {type: data[0] for type, data in model_inputs.items()}