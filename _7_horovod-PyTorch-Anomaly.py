train_len = 295246830
one_len = 1589906
zero_len = 293656924
protein_map = {'BRD4': 1, 'HSA': 2, 'sEH': 3}
vocab = {'C': 6825082866, '#': 81527490, '@': 511451694, 'H': 456489972, '=': 1406606874, 'O': 2554179786,
         'N': 2469595230, 'c': 12257477022, '-': 438483636, '.': 216945504, 'l': 491088828, 'B': 123330132,
         'r': 121915914, 'n': 1997759694, 'D': 295246830, 'y': 295246830, 'o': 67918650, 's': 156618468,
         'S': 90662574, 'F': 492710238, '+': 65206260, 'i': 1414026, '/': 11547096, 'I': 23972994}

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as sF
from pyspark.sql.types import LongType, IntegerType, StructType, StructField

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import numpy as np

import pyspark
import pyspark.sql.types as T
from pyspark import SparkConf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pyspark.sql import SparkSession
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm, trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# SparkSession for 128 GB RAM and 64 cores
spark = (
    SparkSession
    .builder
    .appName("Optimized Spark for 128GB RAM and 64 Cores")
    .config("spark.driver.memory", "64g")  # 64GB for driver memory
    .config("spark.executor.memory", "64g")  # 64GB for executor memory
    .config("spark.executor.instances", "16")  # 16 executors
    .config("spark.executor.cores", "4")  # 4 cores per executor (total = 64 cores)
    .config("spark.driver.maxResultSize", "8g")  # Driver result size limit
    .config("spark.local.dir", "/scratch/23m1521/temp")  # Temp directory with enough space
    .config("spark.shuffle.file.buffer", "512k")  # Increased shuffle buffer size
    .config("spark.memory.fraction", "0.8")  # Memory fraction for tasks
    .config("spark.shuffle.memoryFraction", "0.6")  # Shuffle memory fraction
    .config("spark.executor.javaOptions", "-Xmx64g")  # JVM heap size for executors
    .master("local[64]")  # Use all 64 cores on the machine
    .getOrCreate()
)




df0_features = spark.read.format('parquet').load('zero_features.parquet')
df1_features = spark.read.format('parquet').load('one_features.parquet')

full_df = df0_features.union(df1_features).orderBy(sF.rand())



protein_ohe = OneHotEncoder(inputCol="protein", outputCol="protein_onehot")
protein_ohe = protein_ohe.fit(full_df)

full_df = protein_ohe.transform(full_df)
features_cols = full_df.columns[-1:] + full_df.columns[2:-2]
vectorAssembler = VectorAssembler(inputCols=features_cols, outputCol='features')
full_df2 = vectorAssembler.transform(full_df)


def save_checkpoint(total_steps, epoch, model, optimizer, loss, checkpoint_path="checkpoint.pth"):
    checkpoint = {
        'total_steps': total_steps,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved after {total_steps} steps at epoch {epoch}")

def load_checkpoint(checkpoint_path="checkpoint.pth"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        total_steps = checkpoint['total_steps']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return total_steps, start_epoch, loss
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, 0, None

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(300, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


input_dim = 99
multi_GPU = True

if multi_GPU:
    model = torch.nn.DataParallel(BinaryClassifier(input_dim), device_ids=[0, 1]).to(device)
else:
    model = BinaryClassifier(input_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = int(1024*1)
checkpoint_path = "checkpoints/_2_PyTorch.pth"


def train_partition(features, labels):
    output = model(features)
    loss = criterion(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

total_steps = 0
total_steps, start_epoch, prev_loss = load_checkpoint(checkpoint_path)

for epoch in trange(start_epoch, 10, desc='Epoch', dynamic_ncols=True):
    batch_features = []
    batch_labels = []
    epoch_loss = 0
    steps = 0

    with tqdm(total=int(train_len / batch_size), desc="Training", dynamic_ncols=True) as pbar:
        for row in full_df2.rdd.toLocalIterator():
            batch_features.append(row['features'])
            batch_labels.append(row['y'])

            if len(batch_features) == batch_size:
                features = torch.tensor(batch_features, dtype=torch.float32).to(device)
                labels = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1).to(device)

                loss = train_partition(features, labels)
                epoch_loss += loss
                steps += 1

                batch_features = []
                batch_labels = []

                if steps % 10000 == 0:
                    save_checkpoint(total_steps+steps, epoch, model, 
                                    optimizer, epoch_loss, checkpoint_path)

                pbar.set_description(f"Total Steps: {total_steps+steps}")
                pbar.set_postfix_str(f"Eloss: {epoch_loss / steps} | BLoss: {loss}")
                pbar.update(1)

    epoch_loss /= steps
    total_steps += steps
    print(f"Epoch: {epoch + 1} | Loss: {epoch_loss}")

    save_checkpoint(total_steps, epoch, model, optimizer, epoch_loss, checkpoint_path)