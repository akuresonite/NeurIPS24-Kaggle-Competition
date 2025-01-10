import marimo

__generated_with = "0.10.7"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""## Model Creation on Chunks""")
    return


@app.cell
def _():
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
    from pyspark.sql import functions as F
    from pyspark.sql.types import LongType, IntegerType, StructType, StructField

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    from xgboost.spark import SparkXGBClassifier

    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    return (
        BinaryClassificationEvaluator,
        CrossValidator,
        F,
        IntegerType,
        LogisticRegression,
        LogisticRegressionCV,
        LongType,
        OneHotEncoder,
        ParamGridBuilder,
        SparkSession,
        SparkXGBClassifier,
        StructField,
        StructType,
        VectorAssembler,
        average_precision_score,
        classification_report,
        col,
        np,
        one_len,
        pd,
        protein_map,
        roc_auc_score,
        train_len,
        train_test_split,
        vocab,
        zero_len,
    )


@app.cell
def _(SparkSession):
    # for 256 Gb and 64 Cores
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
    #     .master("local[64]")  # Use all 64 cores on the machine
    #     .getOrCreate()
    # )

    # spark

    spark = (
        SparkSession
        .builder
        .appName("leash belka3")
        .config("spark.driver.memory", "64g")  # Increased driver memory
        .config("spark.executor.memory", "64g")  # Increased executor memory
        .config("spark.executor.instances", "8")  # Reduced number of executors
        .config("spark.executor.cores", "8")  # Increased cores per executor
        .config("spark.driver.maxResultSize", "4g")  # Driver result size limit
        .config("spark.local.dir", "temp")  # Specify a directory with enough space
        .config("spark.shuffle.file.buffer", "128k")  # Shuffle buffer size
        .config("spark.memory.fraction", "0.8")  # Memory fraction for tasks
        .config("spark.shuffle.memoryFraction", "0.7")  # Shuffle memory fraction
        .config("spark.executor.javaOptions", "-Xmx64g")  # JVM heap size for executors
        .config("spark.sql.shuffle.partitions", "1000")  # Increase shuffle partitions
        .config("spark.ui.enabled", "true")  # Enable Spark UI
        .master("local[8]")  # Reduced number of cores for local mode
        .getOrCreate()
    )

    spark
    return (spark,)


@app.cell
def _(F, spark):
    df0_features = spark.read.format('parquet').load('zero_features.parquet')
    df1_features = spark.read.format('parquet').load('one_features.parquet')

    full_df = df0_features.union(df1_features).orderBy(F.rand())

    # print(df0_features.rdd.getNumPartitions())
    # print(full_df.count())
    # df0_features.printSchema()
    return df0_features, df1_features, full_df


@app.cell
def _():
    # sample_df = full_df.sample(fraction=0.00001)
    return


@app.cell
def _(OneHotEncoder_1, full_df):
    from pyspark.ml.feature import OneHotEncoder
    protein_ohe = OneHotEncoder_1(inputCol='protein', outputCol='protein_onehot')
    protein_ohe = protein_ohe.fit(full_df)
    return OneHotEncoder, protein_ohe


@app.cell
def _(full_df, protein_ohe):
    full_df_1 = protein_ohe.transform(full_df)
    return (full_df_1,)


@app.cell
def _(full_df_1):
    features_cols = full_df_1.columns[-1:] + full_df_1.columns[2:-2]
    return (features_cols,)


@app.cell
def _(VectorAssembler, features_cols):
    vectorAssembler = VectorAssembler(inputCols=features_cols, outputCol='features')
    return (vectorAssembler,)


@app.cell
def _(full_df_1, vectorAssembler):
    full_df2 = vectorAssembler.transform(full_df_1)
    return (full_df2,)


@app.cell
def _(SparkXGBClassifier, spark):
    # model = SparkXGBClassifier(num_workers=spark.sparkContext.defaultParallelism, label_col='y')

    xgb_classifier = SparkXGBClassifier(
        label_col="y", 
        features_col="features", 
        num_workers=spark.sparkContext.defaultParallelism,
        # use_gpu=True,
        # device='cuda',
        max_depth=6
    )
    return (xgb_classifier,)


@app.cell
def _(full_df2, xgb_classifier):
    xgb_model = xgb_classifier.fit(full_df2)
    return (xgb_model,)


@app.cell
def _(xgb_model):
    _model_path = 'checkpoints/_1_XGB'
    xgb_model.save(_model_path)
    return


@app.cell
def _():
    model_checkpoint_path = 'checkpoints/_1_XGB'
    from xgboost.spark import SparkXGBClassifierModel
    xgb_model_1 = SparkXGBClassifierModel.load(model_checkpoint_path)
    return SparkXGBClassifierModel, model_checkpoint_path, xgb_model_1


@app.cell
def _(full_df2, xgb_model_1):
    predictions = xgb_model_1.transform(full_df2)
    return (predictions,)


@app.cell
def _(predictions):
    predictions_with_prob = predictions.select("features", "y", "prediction", "probability")
    predictions_with_prob.show()
    return (predictions_with_prob,)


@app.cell
def _(predictions_with_prob):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    # Calculate Accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="y", predictionCol="prediction", metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions_with_prob)
    print(f"Accuracy: {accuracy}")
    return MulticlassClassificationEvaluator, accuracy, accuracy_evaluator


@app.cell
def _(BinaryClassificationEvaluator, predictions_with_prob):
    # Calculate AUC (Area Under ROC)
    auc_evaluator = BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="prediction", metricName="areaUnderROC")
    auc = auc_evaluator.evaluate(predictions_with_prob)
    print(f"AUC: {auc}")
    return auc, auc_evaluator


@app.cell
def _(BinaryClassificationEvaluator, predictions_with_prob):
    # Calculate PR AUC (Area Under Precision-Recall Curve)
    pr_auc_evaluator = BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="prediction", metricName="areaUnderPR")
    pr_auc = pr_auc_evaluator.evaluate(predictions_with_prob)
    print(f"PR AUC (approx MAP): {pr_auc}")
    return pr_auc, pr_auc_evaluator


@app.cell
def _(col, predictions_with_prob):
    TP = predictions_with_prob.filter((col('y') == 1) & (col('prediction') == 1)).count()
    FP = predictions_with_prob.filter((col('y') == 0) & (col('prediction') == 1)).count()
    TN = predictions_with_prob.filter((col('y') == 0) & (col('prediction') == 0)).count()
    FN = predictions_with_prob.filter((col('y') == 1) & (col('prediction') == 0)).count()
    _precision = TP / (TP + FP) if TP + FP > 0 else 0
    print(f'Precision: {_precision}')
    class_counts = predictions_with_prob.groupBy('y').count()
    total_instances = predictions_with_prob.count()
    weighted_precision = 0
    for row in class_counts.collect():
        label = row['y']
        class_count = row['count']
        if label == 1:
            precision_class = TP / (TP + FP) if TP + FP > 0 else 0
        else:
            precision_class = TN / (TN + FN) if TN + FN > 0 else 0
        weight = class_count / total_instances
        weighted_precision = weighted_precision + precision_class * weight
    print(f'Weighted Precision: {weighted_precision}')
    return (
        FN,
        FP,
        TN,
        TP,
        class_count,
        class_counts,
        label,
        precision_class,
        row,
        total_instances,
        weight,
        weighted_precision,
    )


@app.cell
def _(weighted_precision):
    weighted_precision
    return


@app.cell
def _(FN, FP, TN, TP):
    import seaborn as sns
    import matplotlib.pyplot as plt

    print(TP, FP, TN, FN)
    confusion_matrix = [
        [TN, FP],
        [FN, TP]
    ]

    plt.figure(figsize=(5, 3))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    return confusion_matrix, plt, sns


@app.cell
def _():
    (100*155810)/(155810+1434096), (100*1434096)/(155810+1434096)
    return


@app.cell
def _(FN, FP, TP):
    _precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    (_precision, recall)
    return (recall,)


@app.cell
def _():
    """
    To calculate Mean Average Precision (mAP), you need to use the precision at 
    different recall levels. The formula for Average Precision (AP) for each class 
    is the area under the Precision-Recall Curve, which involves precision and 
    recall values calculated at multiple thresholds.
    """
    '\nIn a binary classification task, the mAP is equivalent to the AP since you \nonly have one class. In a multi-class or multi-label task, you would compute \nthe AP for each class and then average them to get the mAP.\n'
    precisions = [0.9, 0.8, 0.7]
    recalls = [0.1, 0.5, 1.0]
    ap = 0
    for i in range(1, len(precisions)):
        recall_diff = recalls[i] - recalls[i - 1]
        ap = ap + precisions[i] * recall_diff
    mAP = ap
    print(f'Mean Average Precision (mAP): {mAP}')
    return ap, i, mAP, precisions, recall_diff, recalls


@app.cell
def _(
    BinaryClassificationEvaluator,
    CrossValidator,
    ParamGridBuilder,
    SparkXGBClassifier,
    best_model_path,
    full_df2,
):
    from pyspark.ml import Pipeline
    xgb_classifier_1 = SparkXGBClassifier(label_Col='y', features_Col='features')
    evaluator = BinaryClassificationEvaluator(labelCol='y', metricName='areaUnderPR')
    param_grid = ParamGridBuilder().addGrid(xgb_classifier_1.max_depth, [3, 5, 7, 10]).addGrid(xgb_classifier_1.learning_rate, [0.01, 0.1, 0.2, 0.3]).addGrid(xgb_classifier_1.subsample, [0.6, 0.8, 1.0]).addGrid(xgb_classifier_1.colsample_bytree, [0.6, 0.8, 1.0]).addGrid(xgb_classifier_1.gamma, [0, 0.1, 0.2, 0.5]).addGrid(xgb_classifier_1.min_child_weight, [1, 3, 5]).addGrid(xgb_classifier_1.reg_lambda, [0, 1, 5]).addGrid(xgb_classifier_1.reg_alpha, [0, 1, 5]).addGrid(xgb_classifier_1.n_estimators, [500, 3000]).build()
    crossval = CrossValidator(estimator=xgb_classifier_1, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    cv_model = crossval.fit(full_df2)
    best_model = cv_model.bestModel
    _model_path = 'checkpoints/_2_XGB_CV'
    best_model.save(best_model_path)
    predictions_1 = best_model.transform(full_df2)
    aucPR = evaluator.evaluate(predictions_1)
    print(f'Test AUC-PR: {aucPR}')
    best_params = cv_model.bestModel.extractParamMap()
    print('Best Hyperparameters:')
    for param, value in best_params.items():
        print(f'{param.name}: {value}')
    return (
        Pipeline,
        aucPR,
        best_model,
        best_params,
        crossval,
        cv_model,
        evaluator,
        param,
        param_grid,
        predictions_1,
        value,
        xgb_classifier_1,
    )


@app.cell
def _(mo):
    mo.md(r"""//////////////////////////////////////////////////////////////////////////////////////////////""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
