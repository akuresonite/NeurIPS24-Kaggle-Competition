import marimo

__generated_with = "0.10.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Model Creation on Chunks
        """
    )
    return


@app.cell
def _():
    train_len = 295246830
    one_len = 1589906
    zero_len = 293656924
    test_len = 1674896
    protein_map = {'BRD4': 1, 'HSA': 2, 'sEH': 3}
    vocab = {'C': 6825082866, '#': 81527490, '@': 511451694, 'H': 456489972, '=': 1406606874, 'O': 2554179786,
             'N': 2469595230, 'c': 12257477022, '-': 438483636, '.': 216945504, 'l': 491088828, 'B': 123330132,
             'r': 121915914, 'n': 1997759694, 'D': 295246830, 'y': 295246830, 'o': 67918650, 's': 156618468,
             'S': 90662574, 'F': 492710238, '+': 65206260, 'i': 1414026, '/': 11547096, 'I': 23972994}

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when
    from pyspark.sql import functions as F
    from pyspark.sql.functions import udf
    from pyspark.sql.types import LongType, IntegerType, StructType, StructField, ArrayType, DoubleType, StringType
    from pyspark.ml.linalg import SparseVector

    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StringIndexerModel, OneHotEncoderModel
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

    import os
    import pandas as pd
    import numpy as np
    import joblib

    from xgboost.spark import SparkXGBClassifier
    from functools import wraps
    import xgboost as xgb

    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

    from concurrent.futures import ThreadPoolExecutor
    from joblib import Parallel, delayed
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdmolops, AllChem, rdchem, rdEHTTools, rdMolDescriptors
    # from tqdm.auto import tqdm
    from tqdm import tqdm
    from padelpy import from_smiles
    from IPython.display import display, HTML
    display(HTML("<style>pre { white-space: pre !important; }</style>"))
    return (
        AllChem,
        ArrayType,
        BinaryClassificationEvaluator,
        Chem,
        CrossValidator,
        Descriptors,
        DoubleType,
        F,
        HTML,
        IntegerType,
        Lipinski,
        LongType,
        MulticlassClassificationEvaluator,
        OneHotEncoder,
        OneHotEncoderModel,
        Parallel,
        ParamGridBuilder,
        SparkSession,
        SparkXGBClassifier,
        SparseVector,
        StringIndexer,
        StringIndexerModel,
        StringType,
        StructField,
        StructType,
        ThreadPoolExecutor,
        VectorAssembler,
        average_precision_score,
        classification_report,
        col,
        delayed,
        display,
        from_smiles,
        joblib,
        np,
        one_len,
        os,
        pd,
        protein_map,
        rdEHTTools,
        rdMolDescriptors,
        rdchem,
        rdmolops,
        roc_auc_score,
        test_len,
        tqdm,
        train_len,
        udf,
        vocab,
        when,
        wraps,
        xgb,
        zero_len,
    )


@app.cell
def _():
    import pyspark
    pyspark.__version__
    return (pyspark,)


@app.cell
def _(SparkSession):
    # for 256 Gb and 64 Cores
    spark = (
        SparkSession
        .builder
        .appName("leash belka3")
        .config("spark.driver.memory", "48g")  # Increased driver memory
        .config("spark.executor.memory", "48g")  # Increased executor memory
        .config("spark.executor.instances", "16")  # 16 executors
        .config("spark.executor.cores", "4")  # 4 cores per executor
        .config("spark.driver.maxResultSize", "4g")  # Driver result size limit
        .config("spark.local.dir", "temp")  # Specify a directory with enough space
        .config("spark.shuffle.file.buffer", "128k")  # Shuffle buffer size
        .config("spark.memory.fraction", "0.8")  # Memory fraction for tasks
        .config("spark.shuffle.memoryFraction", "0.6")  # Shuffle memory fraction
        .config("spark.executor.javaOptions", "-Xmx48g")  # JVM heap size for executors
        .master("local[64]")  # Use all 64 cores on the machine
        .getOrCreate()
    )

    spark

    # --- SynapseML ----------------------------------------
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
    #     .appName("leash belka3467")
    #     .config("spark.driver.memory", "64g")  # Increased driver memory for large jobs
    #     .config("spark.executor.memory", "64g")  # Increased executor memory
    #     .config("spark.executor.instances", "32")  # 32 executors
    #     .config("spark.executor.cores", "2")  # 2 cores per executor
    #     .config("spark.driver.maxResultSize", "8g")  # Driver result size limit
    #     .config("spark.local.dir", "temp")  # Ensure high-speed storage
    #     .config("spark.shuffle.file.buffer", "1024k")  # Larger shuffle buffer for better IO
    #     .config("spark.memory.fraction", "0.85")  # Increased memory for tasks
    #     .config("spark.shuffle.memoryFraction", "0.7")  # Increased shuffle memory
    #     .config("spark.executor.javaOptions", "-Xmx64g")  # JVM heap size for executors
    #     .master("local[*]")  # Use all 64 cores on the machine
    #     .getOrCreate()
    # )
    # spark

    # --- GPU -----------------------------------------------
    # spark = (
    #     SparkSession
    #     .builder
    #     .appName("leash belka3467")
    #     .config("spark.driver.memory", "64g")
    #     .config("spark.executor.memory", "64g")
    #     .config("spark.executor.instances", "2")  # 2 executors, one per GPU
    #     .config("spark.executor.cores", "32")  # Divide cores equally between executors (64/2)
    #     .config("spark.driver.maxResultSize", "8g")
    #     .config("spark.local.dir", "temp")
    #     .config("spark.shuffle.file.buffer", "1024k")
    #     .config("spark.memory.fraction", "0.85")
    #     .config("spark.shuffle.memoryFraction", "0.7")
    #     .config("spark.executor.javaOptions", "-Xmx64g")
    #     .config("spark.executor.resource.gpu.amount", "1") # Assign 1 GPU per executor
    #     .config("spark.master", "local[*]") # Important: Use local cluster mode to enable GPU scheduling
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
    return (spark,)


@app.cell
def _():
    datadir = "/home/23m1521/ashish/kaggle/full_feat_tok_df_vectors.parquet"
    return (datadir,)


@app.cell
def _(col, spark, when):
    def load_df_chunk(path):
        return spark.read.format('parquet').load(path)

    def add_sample_weights(df):
        class_counts = df.groupBy("binds").count().collect()
        total_count = sum(row["count"] for row in class_counts)
        class_weights = {row["binds"]: total_count / (2 * row["count"]) for row in class_counts}
        return df.withColumn("sample_weights", when(col("binds") == 0, class_weights[0]).when(col("binds") == 1, class_weights[1]))

    def get_scale_pos_weight(df):
        class_counts = dict(df.groupBy("binds").count().collect())
        return class_counts[0]/class_counts[1]
    return add_sample_weights, get_scale_pos_weight, load_df_chunk


@app.cell
def _(add_sample_weights, datadir, load_df_chunk):
    # --- Load chunk --------------------------------------
    full_df = load_df_chunk(datadir)
    full_df = add_sample_weights(full_df)
    full_df = full_df.repartition(1000)
    full_df_count = full_df.count()
    print(full_df_count)
    full_df.show()
    return full_df, full_df_count


@app.cell
def _(full_df):
    print(full_df.rdd.getNumPartitions())
    return


@app.cell
def _():
    import optuna
    STUDY_NAME = f"XGB_HPC"

    loaded_study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///db_{STUDY_NAME}.sqlite3",
    )

    Best_Parameters_lite = loaded_study.get_trials()[95].params
    print(Best_Parameters_lite)
    return Best_Parameters_lite, STUDY_NAME, loaded_study, optuna


@app.cell
def _(Best_Parameters_lite, SparkXGBClassifier):
    Best_Parameters = {
        'max_depth': 17, 
        'eta': 0.05, 
        'subsample': 1.0, 
        'colsample_bytree': 0.6, 
        'gamma': 0.1, 
        'min_child_weight': 10, 
        'lambda': 0, 
        'alpha': 5, 
        'n_estimators': 3000
    }

    xgb_classifier = SparkXGBClassifier(
        features_col="vectors", 
        label_col="binds",
        weight_col="sample_weights",
        # num_workers=spark.sparkContext.defaultParallelism,
        num_workers=1,
        # use_gpu=True,
        # device='cuda',
        eval_metric='aucpr',
        **Best_Parameters_lite
    )
    return Best_Parameters, xgb_classifier


@app.cell
def _(full_df, xgb_classifier):
    xgb_model = xgb_classifier.fit(full_df)
    return (xgb_model,)


@app.cell
def _(xgb_model):
    _model_path = 'checkpoints/_4_XGB_Feat_tok_GPU_CV'
    xgb_model.save(_model_path)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
