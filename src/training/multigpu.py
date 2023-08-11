import pickle
import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from datautils import AutoCorrectionDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: unique identifier of each process
        world_size: total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, input_ids, attention_mask, labels):
        self.optimizer.zero_grad()
        output = self.model(input_ids, labels=labels, attention_mask = attention_mask)

        loss = output.loss
        average_loss = loss.item()
        loss.backward()

        self.optimizer.step()
        
        return average_loss

    def _run_epoch(self, epoch):
        b_sz = next(iter(self.train_data))['input_ids'].shape[0]

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

        self.train_data.sampler.set_epoch(epoch)

        average_loss = 0

        for batch in tqdm(self.train_data):
            params = {}

            for property in ['input_ids', 'labels', 'attention_mask']:
                params[property] = batch[property].to(self.gpu_id)

            average_loss += self._run_batch(**params)
        
        print(f'Epoch number {epoch} run with {average_loss/len(self.train_data)} loss.')

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

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

def load_dataset():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    annotated_data = datasets.load_dataset("carolmou/random-sentences")["train"]
        
    wrong = annotated_data['wrong_text']
    correct = annotated_data['correct_text']


    train_data = [tup for tup in zip(wrong, correct)]

    train_dataset = AutoCorrectionDataset(train_data, tokenizer, 128)

    return train_dataset


def load_train_objs():
    dataset = load_dataset()
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.load_state_dict(torch.load("checkpoint.pt"))
    optimizer = AdamW(model.parameters(), lr=1e-5)
    return dataset,model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset,model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)