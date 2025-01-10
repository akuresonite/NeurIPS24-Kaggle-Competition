train_len = 295246830
one_len = 1589906
zero_len = 293656924
protein_map = {'BRD4': 1, 'HSA': 2, 'sEH': 3}
vocab = {'C': 6825082866, '#': 81527490, '@': 511451694, 'H': 456489972, '=': 1406606874, 'O': 2554179786,
         'N': 2469595230, 'c': 12257477022, '-': 438483636, '.': 216945504, 'l': 491088828, 'B': 123330132,
         'r': 121915914, 'n': 1997759694, 'D': 295246830, 'y': 295246830, 'o': 67918650, 's': 156618468,
         'S': 90662574, 'F': 492710238, '+': 65206260, 'i': 1414026, '/': 11547096, 'I': 23972994}

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

import os
import numpy as np
import joblib

from functools import wraps
import xgboost as xgb
from tqdm.auto import tqdm
# from tqdm import tqdm
from padelpy import from_smiles
# from IPython.display import display, HTML
# display(HTML("<style>pre { white-space: pre !important; }</style>"))


# # for 256 Gb and 64 Cores
# spark = (
#     SparkSession
#     .builder
#     .appName("leash belka3")
#     .config("spark.driver.memory", "48g")  # Increased driver memory
#     .config("spark.executor.memory", "48g")  # Increased executor memory
#     .config("spark.executor.instances", "16")  # 16 executors
#     .config("spark.executor.cores", "4")  # 4 cores per executor
#     .config("spark.driver.maxResultSize", "4g")  # Driver result size limit
#     .config("spark.local.dir", "temp")  # Specify a directory with enough space
#     # .config("spark.local.dir", "/scratch/23m1521/temp")  # Specify a directory with enough space
#     .config("spark.shuffle.file.buffer", "128k")  # Shuffle buffer size
#     .config("spark.memory.fraction", "0.8")  # Memory fraction for tasks
#     .config("spark.shuffle.memoryFraction", "0.6")  # Shuffle memory fraction
#     .config("spark.executor.javaOptions", "-Xmx48g")  # JVM heap size for executors
#     .master("local[64]")  # Use all 64 cores on the machine
#     .getOrCreate()
# )

# spark
spark = (
    SparkSession
    .builder
    .appName("leash belka3467")
    .config("spark.driver.memory", "64g")  # Increased driver memory for large jobs
    .config("spark.executor.memory", "64g")  # Increased executor memory
    .config("spark.executor.instances", "32")  # 32 executors
    .config("spark.executor.cores", "2")  # 2 cores per executor
    .config("spark.driver.maxResultSize", "8g")  # Driver result size limit
    .config("spark.local.dir", "temp")  # Ensure high-speed storage
    .config("spark.shuffle.file.buffer", "1024k")  # Larger shuffle buffer for better IO
    .config("spark.memory.fraction", "0.85")  # Increased memory for tasks
    .config("spark.shuffle.memoryFraction", "0.7")  # Increased shuffle memory
    .config("spark.executor.javaOptions", "-Xmx64g")  # JVM heap size for executors
    .master("local[*]")  # Use all 64 cores on the machine
    .getOrCreate()
)
spark

# SparkSession for 128 GB RAM and 64 cores
# spark = (
#     SparkSession
#     .builder
#     .appName("Optimized Spark for 128GB RAM and 64 Cores")
#     .config("spark.driver.memory", "64g")  # 64GB for driver memory
#     .config("spark.executor.memory", "64g")  # 64GB for executor memory
#     .config("spark.executor.instances", "16")  # 16 executors
#     .config("spark.executor.cores", "4")  # 4 cores per executor (total = 64 cores)
#     .config("spark.driver.maxResultSize", "8g")  # Driver result size limit
#     .config("spark.local.dir", "temp")  # Temp directory with enough space
#     .config("spark.shuffle.file.buffer", "512k")  # Increased shuffle buffer size
#     .config("spark.memory.fraction", "0.8")  # Memory fraction for tasks
#     .config("spark.shuffle.memoryFraction", "0.6")  # Shuffle memory fraction
#     .config("spark.executor.javaOptions", "-Xmx64g")  # JVM heap size for executors
#     .master("local[64]")  # Use all 64 cores on the machine
#     .getOrCreate()
# )

# spark

# SynapseML 
# spark = (
#     SparkSession
#     .builder
#     .appName("leash belka3")
#     .config("spark.driver.memory", "48g")  # Increased driver memory
#     .config("spark.executor.memory", "48g")  # Increased executor memory
#     .config("spark.executor.instances", "16")  # 16 executors
#     .config("spark.executor.cores", "4")  # 4 cores per executor
#     .config("spark.driver.maxResultSize", "4g")  # Driver result size limit
#     .config("spark.local.dir", "temp")  # Specify a directory with enough space
#     .config("spark.shuffle.file.buffer", "128k")  # Shuffle buffer size
#     .config("spark.memory.fraction", "0.8")  # Memory fraction for tasks
#     .config("spark.shuffle.memoryFraction", "0.6")  # Shuffle memory fraction
#     .config("spark.executor.javaOptions", "-Xmx48g")  # JVM heap size for executors
#     .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.8")
#     .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
#     .master("local[64]")  # Use all 64 cores on the machine
#     .getOrCreate()
# )

# spark

# spark = (
#     SparkSession
#     .builder
#     .appName("leash belka3")
#     .config("spark.driver.memory", "64g")  # Increased driver memory
#     .config("spark.executor.memory", "64g")  # Increased executor memory
#     .config("spark.executor.instances", "8")  # Reduced number of executors
#     .config("spark.executor.cores", "8")  # Increased cores per executor
#     .config("spark.driver.maxResultSize", "4g")  # Driver result size limit
#     .config("spark.local.dir", "temp")  # Specify a directory with enough space
#     .config("spark.shuffle.file.buffer", "128k")  # Shuffle buffer size
#     .config("spark.memory.fraction", "0.8")  # Memory fraction for tasks
#     .config("spark.shuffle.memoryFraction", "0.7")  # Shuffle memory fraction
#     .config("spark.executor.javaOptions", "-Xmx64g")  # JVM heap size for executors
#     .config("spark.sql.shuffle.partitions", "1000")  # Increase shuffle partitions
#     .config("spark.ui.enabled", "true")  # Enable Spark UI
#     .master("local[8]")  # Reduced number of cores for local mode
#     .getOrCreate()
# )

# spark



datadir = "/home/23m1521/ashish/kaggle/full_feat_tok_df_vectors.parquet"
chunks_path = sorted([os.path.join(datadir, i) for i in os.listdir(datadir) if i.endswith(".parquet")])
total_chunks = len(chunks_path)
print(total_chunks)




def load_df_chunk(path):
    return spark.read.format('parquet').load(path)

def get_scale_pos_weight(df):
    class_counts = dict(df.groupBy("binds").count().collect())
    return class_counts[0]/class_counts[1]

def add_sample_weights(df):
    class_counts = df.groupBy("binds").count().collect()
    total_count = sum(row["count"] for row in class_counts)
    class_weights = {row["binds"]: total_count / (2 * row["count"]) for row in class_counts}
    return df.withColumn("sample_weights", when(col("binds") == 0, class_weights[0]).when(col("binds") == 1, class_weights[1]))

def make_dataset(df, chunk_df_count):
    def process_row(row):
        return (row['vectors'].toArray(), row['binds'], row['sample_weights'])
    features, labels, weights = [], [], []
    for feature, label, weight in tqdm(df.rdd.map(process_row).toLocalIterator(), total=chunk_df_count):
        features.append(feature)
        labels.append(label)
        weights.append(weight)
    return features, labels, weights

def make_dataset2(df, test=False):
    df = df.toPandas()
    df.vectors = df.vectors.map(lambda x: x.toArray())
    if test == True:
        return df.id.values, np.array([i for i in df.vectors.values]), np.empty_like(df.id.values), None
    return df.id.values, np.array([i for i in df.vectors.values]), df.binds.values, df.sample_weights.values


def save_checkpoint(model, params, i, evals_result,  path, save_name):
    os.makedirs(path, exist_ok=True)
    model.save_model(os.path.join(path, f"{save_name}.json"))
    joblib.dump({"params": params, 'i': i, "evals_result": evals_result}, os.path.join(path, f"{save_name}_params.joblib"))
    print("Model saved at", path)

def load_checkpoint(path, save_name):
    model = xgb.Booster()
    model.load_model(os.path.join(path, f"{save_name}.json"))
    ckpt = joblib.load(os.path.join(path, f"{save_name}_params.joblib"))
    params, i = ckpt['params'], ckpt['i']
    print("Model loaded from", path)
    return model, params, i

def train_xgb(dmatrix, xgb_model=None):
    lr = [0.1, 0.07, 0.04, 0.01, 0.007]
    best_params1 = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'aucpr'],
    
    'max_depth': 13, 
    'eta': 0.2, 
    'subsample': 1.0, 
    'colsample_bytree': 0.5, 
    'gamma': 0.5, 
    'min_child_weight': 7, 
    'lambda': 5,
    'alpha': 5,
    'num_boost_round': 100,
    
    'rate_drop': 0.4,
    'skip_drop': 0.5,
    'seed': 42,
    'device': 'cuda'
    }
    evals_result = {'train': {'logloss':[], 'aucpr': []}}
    
    bst = xgb.train(
        best_params1,
        dtrain=dmatrix, 
        num_boost_round=100,
        evals=[(dmatrix, 'train')], 
        evals_result=evals_result, 
        verbose_eval=False,
        xgb_model=xgb_model
        # early_stopping_rounds=500,
        )
    return bst, evals_result, best_params1

def delete_df_chunk(df):
    df.unpersist()
    del df

def spark_suppress_logs(level="ERROR", reset_level="INFO"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            spark.sparkContext.setLogLevel(level)
            try:
                return func(*args, **kwargs)
            finally:
                spark.sparkContext.setLogLevel(reset_level)
        return wrapper
    return decorator


# @spark_suppress_logs()
def incrementally_train():
    with tqdm(total=total_chunks, dynamic_ncols=True) as pbar1:
        sample_count = 0
        xgb_model = None
        ckpt_dir = "Incrementally_train_XGB2_ckpt"
        
        for i, chunk_path in enumerate(chunks_path):
            # if i > 1:
            #     break
        
            pbar1.set_description(f"Chunk: {i+1}")
            
            
            # --- Load chunk --------------------------------------
            chunk_df = load_df_chunk(chunk_path)
            chunk_df = add_sample_weights(chunk_df)
            chunk_df = chunk_df.repartition(1)
            chunk_df_count = chunk_df.count()
            
            # --- Getting Dataset ---------------------------------
            features, labels, weights = make_dataset2(chunk_df)
            # dtrain = xgb.DMatrix(data=features, label=labels, weight=weights, nthread=25)
            dtrain = xgb.DMatrix(data=features, label=labels, nthread=25)
            
            # # --- Train ------------------------------------------
            if xgb_model is None:
                xgb_model, evals_result, params = train_xgb(dtrain)
            else:
                xgb_model, evals_result, params = train_xgb(dtrain, xgb_model)
            save_checkpoint(xgb_model, params, i, evals_result, ckpt_dir, f"_{i+1}_ckpt")
            
            # # --- Model Evaluation -------------------------------
            aucpr = np.mean(evals_result['train']['aucpr'])
            print(f"Chunk {i+1} trained. AUCPR: {aucpr}")
            
            
            # --- Clean up ----------------------------------------
            # delete_df_chunk(chunk_df)
            del chunk_df, features, labels, weights, dtrain
            sample_used = sample_count + chunk_df_count
            sample_used_percentage = (sample_used / train_len) * 100
            remaining_samples = train_len - sample_used
            remaining_samples_percentage = (remaining_samples / train_len) * 100
            pbar1.set_postfix_str(
                f"{sample_used} ({sample_used_percentage:.2f}%) samples used," 
                f"{remaining_samples} ({remaining_samples_percentage:.2f}%) samples remaining"
            )
            pbar1.update(1)
            sample_count += chunk_df_count
            
        # --- Save final model -----------------------------------
        save_checkpoint(xgb_model, params, i, evals_result, ckpt_dir, f"Final_ckpt")


if __name__ == "__main__":
    spark.sparkContext.setLogLevel("ERROR")
    incrementally_train()
    spark.sparkContext.setLogLevel("INFO")