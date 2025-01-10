from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifier

# Create a SparkSession
spark = SparkSession.builder.appName("LoadXGBoostModel").getOrCreate()

# Path to the XGBoost model checkpoint
ckpt = "Incrementally_train_XGB_ckpt/_45_ckpt.json"

# Load the XGBoost model from the checkpoint
spark_model = SparkXGBClassifier.load(ckpt)

# Print the loaded model (for inspection)
print(spark_model)

# Stop the SparkSession
spark.stop()
