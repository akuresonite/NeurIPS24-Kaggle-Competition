import pandas as pd
import numpy as np
import os

import dask
# dask.config.set(scheduler='threads')
import dask.dataframe as dd

from collections import Counter
import re

import joblib
from tqdm.auto import trange, tqdm
from IPython.display import display

# import torch
# from torch.amp import autocast
# from torch.utils.data import IterableDataset, DataLoader, Dataset
# import torch.nn as nn
# import torch.optim as optim
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# print(device)

protein_map = {'BRD4': 1, 'HSA': 2, 'sEH': 3}



# def make_vocab(dff, update=None):
#     letter_counts = Counter(update) if update else Counter()
#     l = dff.drop(columns=['id', 'protein_name', 'binds']).to_numpy().flatten()
#     for text in tqdm(l, desc='making vocab'):
#         text = re.sub(r'[\d()\[\]{}]+', '', text)
#         # letter_counts.update(char for char in text)
#         letter_counts.update(text)
#     return dict(letter_counts)
# Optimized Functions
def make_vocab(dff, update=None):
    letter_counts = Counter(update) if update else Counter()
    l = dff.drop(columns=['id', 'protein_name', 'binds']).to_numpy().flatten()
    l = np.char.replace(l, r'[\d()\[\]{}]+', '', regex=True)
    letter_counts.update(''.join(l))
    return dict(letter_counts)

# def make_counter(l):
#     letter_counts = Counter()
#     for text in l:
#         text = re.sub(r'[\d()\[\]{}]+', '', text)
#         letter_counts.update(char for char in text)
#     return dict(letter_counts)

# def allign_counter_to_vocab(counter, vocab):
#     temp = {}
#     for i in range(len(vocab.keys())):
#         if list(vocab.keys())[i] in counter.keys():
#             temp[list(vocab.keys())[i]] = counter[list(vocab.keys())[i]]
#         else:
#             temp[list(vocab.keys())[i]] = 0
#     return temp
def make_counter(l):
    l = re.sub(r'[\d()\[\]{}]+', '', ''.join(l))
    return dict(Counter(l))

def allign_counter_to_vocab(counter, vocab):
    return {key: counter.get(key, 0) for key in vocab.keys()}

def make_features(df, vocab):
    id = df['id'].to_numpy()
    smiles = df.drop(columns=['id', 'protein_name', 'binds']).to_numpy()
    protein = df['protein_name'].to_numpy()
    y = df['binds'].to_numpy()

    df_features = {'id':[], 'bb1':[], 'bb2':[], 'bb3':[], 'molecule':[], 'protein':[], 'y':[]}
    for i in trange(len(id), desc='making features'):
        df_features['id'].append(id[i])

        counter = make_counter(smiles[i][0])
        df_features['bb1'].append(allign_counter_to_vocab(counter, vocab))

        counter = make_counter(smiles[i][1])
        df_features['bb2'].append(allign_counter_to_vocab(counter, vocab))

        counter = make_counter(smiles[i][2])
        df_features['bb3'].append(allign_counter_to_vocab(counter, vocab))

        counter = make_counter(smiles[i][3])
        df_features['molecule'].append(allign_counter_to_vocab(counter, vocab))

        df_features['protein'].append(protein[i])
        df_features['y'].append(y[i])

    return df_features

def check_df_allignment(dff_features, vocab):
    flag = True
    for i in trange(len(dff_features['bb1'])):
        if dff_features['bb1'][i].keys() != vocab.keys():
            print(dff_features['bb1'][i].keys())
            print(vocab.keys())
            flag = False
            break
    return flag


def df_vectors(dff_features, vocab, protein_map):
    op = np.empty((100,7))
    for i in trange(0,len(dff_features['id']),100, desc='Making vector df'):
        df = pd.DataFrame({
            'id': dff_features['id'][i:i+100],
            'bb1': dff_features['bb1'][i:i+100],
            'bb2': dff_features['bb2'][i:i+100],
            'bb3': dff_features['bb3'][i:i+100],
            'molecule': dff_features['molecule'][i:i+100],
            'protein': dff_features['protein'][i:i+100],
            'y': dff_features['y'][i:i+100]
        })

        df.bb1 = df.bb1.apply(lambda x: list(x.values()))
        df.bb2 = df.bb2.apply(lambda x: list(x.values()))
        df.bb3 = df.bb3.apply(lambda x: list(x.values()))
        df.molecule = df.molecule.apply(lambda x: list(x.values()))
        df.protein = df.protein.map(protein_map)

        op = np.concatenate((op, df.to_numpy()))

    return op[100:]


def process_row(row, protein_map=protein_map):
    return {
             'id': row['id'],
             'bb1': list(allign_counter_to_vocab(make_counter(row['buildingblock1_smiles']), vocab).values()),
             'bb2': list(allign_counter_to_vocab(make_counter(row['buildingblock2_smiles']), vocab).values()),
             'bb3': list(allign_counter_to_vocab(make_counter(row['buildingblock3_smiles']), vocab).values()),
             'molecule': list(allign_counter_to_vocab(make_counter(row['molecule_smiles']), vocab).values()),
             'protein': protein_map[row['protein_name']],
             'y': row['binds']
        }

def split(path, frac):
    dask_df = dd.read_parquet(path)
    train_fraction = frac
    train_df, val_df = dask_df.random_split([train_fraction, 1 - train_fraction], random_state=42)
    print(f"Train size: {train_df.shape[0].compute()}")
    print(f"Validation size: {val_df.shape[0].compute()}")
    train_df.to_parquet("train_split.parquet", write_index=False)
    val_df.to_parquet("val_split.parquet", write_index=False)

def split2(path, frac=0.5):
    dask_df = dd.read_parquet(path)
    train_fraction = frac

    f1, f2 = dask_df.random_split([train_fraction, 1 - train_fraction], random_state=42)
    f3, f4 = f1.random_split([train_fraction, 1 - train_fraction], random_state=42)
    f5, f6 = f2.random_split([train_fraction, 1 - train_fraction], random_state=42)

    f7, f8 = f3.random_split([train_fraction, 1 - train_fraction], random_state=42)
    f9, f10 = f4.random_split([train_fraction, 1 - train_fraction], random_state=42)
    f11, f12 = f5.random_split([train_fraction, 1 - train_fraction], random_state=42)
    f13, f14 = f6.random_split([train_fraction, 1 - train_fraction], random_state=42)

    print(f"Split-1: {f7.shape[0].compute()}")
    f7.to_parquet("Train_Full_Split-1.parquet", write_index=False)
    print(f"Split-2: {f8.shape[0].compute()}")
    f8.to_parquet("Train_Full_Split-2.parquet", write_index=False)
    print(f"Split-3: {f9.shape[0].compute()}")
    f9.to_parquet("Train_Full_Split-3.parquet", write_index=False)
    print(f"Split-4: {f10.shape[0].compute()}")
    f10.to_parquet("Train_Full_Split-4.parquet", write_index=False)
    print(f"Split-5: {f11.shape[0].compute()}")
    f11.to_parquet("Train_Full_Split-5.parquet", write_index=False)
    print(f"Split-6: {f12.shape[0].compute()}")
    f12.to_parquet("Train_Full_Split-6.parquet", write_index=False)
    print(f"Split-7: {f13.shape[0].compute()}")
    f13.to_parquet("Train_Full_Split-7.parquet", write_index=False)
    print(f"Split-8: {f14.shape[0].compute()}")
    f14.to_parquet("Train_Full_Split-8.parquet", write_index=False)



vocab = {'C': 6825082866, '#': 81527490, '@': 511451694, 'H': 456489972, '=': 1406606874, 'O': 2554179786, 'N': 2469595230, 'c': 12257477022, '-': 438483636, '.': 216945504, 'l': 491088828, 'B': 123330132, 'r': 121915914, 'n': 1997759694, 'D': 295246830, 'y': 295246830, 'o': 67918650, 's': 156618468, 'S': 90662574, 'F': 492710238, '+': 65206260, 'i': 1414026, '/': 11547096, 'I': 23972994}



class ParquetDataset(IterableDataset):
    def __init__(self, dask_df, vocab=vocab, protein_map=protein_map, transform=None):
        self.dask_df = dask_df
        self.partitions = self.dask_df.to_delayed()
        self.vocab = vocab
        self.protein_map = protein_map
        self.transform = transform
        

    def __iter__(self):
        for partition in self.partitions:
            chunk = partition.compute()
            for _, row in chunk.iterrows():
                yield self.process_row(row)

    def process_row(self, row):
        data = {
            'id': row['id'],
            'bb1': list(allign_counter_to_vocab(make_counter(row['buildingblock1_smiles']), self.vocab).values()),
            'bb2': list(allign_counter_to_vocab(make_counter(row['buildingblock2_smiles']), self.vocab).values()),
            'bb3': list(allign_counter_to_vocab(make_counter(row['buildingblock3_smiles']), self.vocab).values()),
            'molecule': list(allign_counter_to_vocab(make_counter(row['molecule_smiles']), self.vocab).values()),
            'protein': self.protein_map[row['protein_name']],
            'y': row['binds']
        }
        if self.transform:
            data = self.transform(data)
        return data


def custom_collate_fn(batch):
    ids = [sample['id'] for sample in batch]
    bb1 = torch.tensor([sample['bb1'] for sample in batch], dtype=torch.float32)
    bb2 = torch.tensor([sample['bb2'] for sample in batch], dtype=torch.float32)
    bb3 = torch.tensor([sample['bb3'] for sample in batch], dtype=torch.float32)
    molecule = torch.tensor([sample['molecule'] for sample in batch], dtype=torch.float32)
    protein = torch.tensor([sample['protein'] for sample in batch], dtype=torch.long)
    y = torch.tensor([sample['y'] for sample in batch], dtype=torch.float32)

    combined_features = torch.cat([bb1, bb2, bb3, molecule, protein.unsqueeze(1)], dim=1)

    return (
        ids,
        combined_features,
        y
    )



from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
import pandas as pd
import os
import gc  # For garbage collection

# Load Dask DataFrame
dask_df = dd.read_parquet("/home/23m1521/ashish/kaggle/train.parquet")

df_len = dask_df.shape[0].compute()
print(f"Number of rows: {df_len}")

df_dataset = ParquetDataset(dask_df)

# Output directory for chunk files
output_dir = 'chunks_output'
os.makedirs(output_dir, exist_ok=True)

chunk_size = 1000000  # Size of each chunk
chunk_data = []

with Progress(
    "[cyan]{task.description}",
    BarColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Processing...", total=df_len)
    
    for i, data in enumerate(df_dataset):
        progress.update(task, advance=1)
        chunk_data.append(data)
        
        if (i + 1) % chunk_size == 0:
            # Save the chunk to a separate parquet file
            chunk_file = os.path.join(output_dir, f"chunk_{(i + 1) // chunk_size}.parquet")
            df = pd.DataFrame(chunk_data)
            df.to_parquet(chunk_file, engine='pyarrow', compression='snappy', index=False)
            print(f"Saved {chunk_file}")
            
            # Free RAM
            del chunk_data, df
            chunk_data = []
            gc.collect()

# Save remaining data
if chunk_data:
    chunk_file = os.path.join(output_dir, f"chunk_{(df_len // chunk_size) + 1}.parquet")
    df = pd.DataFrame(chunk_data)
    df.to_parquet(chunk_file, engine='pyarrow', compression='snappy', index=False)
    print(f"Saved {chunk_file}")
    
    # Free RAM
    del chunk_data, df
    gc.collect()


# dask_df = dd.read_parquet("/home/23m1521/ashish/kaggle/train.parquet")

# df_len = dask_df.shape[0].compute()
# print(f"Number of rows: {df_len}")


# df_dataset = ParquetDataset(dask_df)


# output_file = 'full_train_dataset.parquet'
# chunk_size = 1000000
# chunk_data = []

# file_exists = os.path.exists(output_file)

# # for i, data in tqdm(enumerate(df_dataset), total=df_len, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
# #     chunk_data.append(data)
    
# #     if (i + 1) % chunk_size == 0:
# #         df = pd.DataFrame(chunk_data)
        
# #         if file_exists:
# #             existing_df = pd.read_parquet(output_file, engine='pyarrow')
# #             updated_df = pd.concat([existing_df, df], ignore_index=True)
# #             updated_df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
# #         else:
# #             df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
# #             file_exists = True
        
# #         file_size = os.path.getsize(output_file) / (1024 * 1024)
# #         print(f"File size after saving chunk {(i + 1) // chunk_size}: {file_size:.2f} MB")
        
# #         chunk_data = []
# from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
# import pandas as pd
# import os

# chunk_data = []
# file_exists = False

# with Progress(
#     "[cyan]{task.description}",
#     BarColumn(),
#     TaskProgressColumn(),
#     MofNCompleteColumn(),
#     TimeElapsedColumn(),
#     TimeRemainingColumn(),
# ) as progress:
#     task = progress.add_task("Processing...", total=df_len)
    
#     for i, data in enumerate(df_dataset):
#         progress.update(task, advance=1)
#         chunk_data.append(data)
        
#         if (i + 1) % chunk_size == 0:
#             df = pd.DataFrame(chunk_data)
            
#             if file_exists:
#                 existing_df = pd.read_parquet(output_file, engine='pyarrow')
#                 updated_df = pd.concat([existing_df, df], ignore_index=True)
#                 updated_df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
#             else:
#                 df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
#                 file_exists = True
            
#             file_size = os.path.getsize(output_file) / (1024 * 1024)
#             print(f"File size after saving chunk {(i + 1) // chunk_size}: {file_size:.2f} MB")
            
#             chunk_data = []



# if chunk_data:
#     df = pd.DataFrame(chunk_data)
#     if file_exists:
#         existing_df = pd.read_parquet(output_file, engine='pyarrow')
#         updated_df = pd.concat([existing_df, df], ignore_index=True)
#         updated_df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
#     else:
#         df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
    
#     file_size = os.path.getsize(output_file) / (1024 * 1024)
#     print(f"Final file size: {file_size:.2f} MB")

