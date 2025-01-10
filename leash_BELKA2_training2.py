import pandas as pd
import numpy as np
import os
import gc
import dask
from multiprocessing.pool import Pool
import dask.dataframe as dd
from collections import Counter
import re
from tqdm.auto import tqdm, trange

import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_len = 295246830
protein_map = {'BRD4': 1, 'HSA': 2, 'sEH': 3}
vocab = {'C': 6825082866, '#': 81527490, '@': 511451694, 'H': 456489972, '=': 1406606874, 'O': 2554179786, 
         'N': 2469595230, 'c': 12257477022, '-': 438483636, '.': 216945504, 'l': 491088828, 'B': 123330132, 
         'r': 121915914, 'n': 1997759694, 'D': 295246830, 'y': 295246830, 'o': 67918650, 's': 156618468, 
         'S': 90662574, 'F': 492710238, '+': 65206260, 'i': 1414026, '/': 11547096, 'I': 23972994}
datadir = '/home/23m1521/ashish/kaggle/chunks_output'



class ProcessedDataset(IterableDataset):
    def __init__(self, datadir, vocab, protein_map):
        self.chunks_path = [os.path.join(datadir, i) for i in os.listdir(datadir)]
        self.vocab = vocab
        self.protein_map = protein_map

    def __iter__(self):
        for chunk_path in self.chunks_path:
            dask_df = dd.read_parquet(path=chunk_path, engine='pyarrow')
            for batch in dask_df.partitions:
                chunk = batch.compute()
                for _, row in chunk.iterrows():
                    yield self.process_row(row)
                del chunk
                gc.collect()

    def process_row(self, row):
        features_list = [
            np.array(row['bb1']).flatten(),
            np.array(row['bb2']).flatten(),
            np.array(row['bb3']).flatten(),
            np.array(row['molecule']).flatten(),
            np.array(row['protein']).flatten()
        ]
        
        features_array = np.concatenate(features_list).astype(np.float32)
        features = torch.tensor(features_array, dtype=torch.float32)
    
        data = {
            'id': torch.tensor(row['id'], dtype=torch.long),
            'features': features,
            'y': torch.tensor(row['y'], dtype=torch.float32)
        }
        return data

def collate_fn(batch):
    ids = torch.stack([item['id'] for item in batch])
    features = torch.stack([item['features'] for item in batch])
    y = torch.stack([item['y'] for item in batch])
    return {'id': ids, 'features': features, 'y': y}


dataset = ProcessedDataset(datadir, vocab, protein_map)
batch_size = int(1024*100)
if device == 'cuda':
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=False, num_workers=20)
else:
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=False, num_workers=10)

n_train_batchs = int(train_len/batch_size)
n_val_batchs = int(train_len/batch_size)
print(n_train_batchs, n_train_batchs)


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Sequential(
            
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(200, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train_model(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    device=device, 
    num_epochs=10, 
    accumulation_steps=4, 
    checkpoint_path="checkpoint.pth"
):

    metrics = {"train_loss": [], "train_accuracy": []}
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}...")

    for epoch in trange(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        batch_count = 0
        row_count = 0
        
        with tqdm(total=n_train_batchs, desc='Training') as pbar:
            for batch_idx, (batch) in enumerate(train_loader):
                inputs = batch['features'].to(device)
                labels = batch['y'].to(device)
                with autocast(device):
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                total_loss += loss.item()
                batch_count += 1

                loss.backward()
                
                predictions = (outputs.sigmoid() > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                row_count += labels.shape[0]
                pbar.set_description(f"Training | {batch_count*batch_size} | {total}")
                pbar.update(1)
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    del inputs, labels, outputs
                    gc.collect()
                
                del batch
                gc.collect()

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at epoch {epoch+1}")
        
        # train_loss = total_loss / batch_count
        # train_accuracy = correct / total

        # metrics["train_loss"].append(train_loss)
        # metrics["train_accuracy"].append(train_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        # print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    del train_loader
    gc.collect()
    return metrics




model = BinaryClassifier(97).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)


train_model(
    model=model, 
    train_loader=train_dataloader, 
    criterion=criterion, 
    optimizer=optimizer,
    device=device,
    num_epochs=10,
    accumulation_steps=10
)