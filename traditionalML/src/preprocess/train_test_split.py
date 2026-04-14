import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import warnings
warnings.filterwarnings("ignore")
import logging
import os, sys
from datetime import datetime
import pytz
vntz = pytz.timezone('Asia/Ho_Chi_Minh')

ROOT_DIR = r""
sys.path.insert(0,ROOT_DIR)
from src.utils.helper_func import load_config, save_config, create_spark_session

LOG_DIR = os.path.join(ROOT_DIR, "artifacts", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout), 
              logging.FileHandler(os.path.join(LOG_DIR, f"train_test_split_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}.log"))]
)

logger = logging.getLogger(__name__)

START_POINT = ""
SPLIT_POINT = ""
COT_POINT = ""

SPLIT_DIR = os.path.join(ROOT_DIR, "data", "split_data")
os.makedirs(SPLIT_DIR, exist_ok=True)

def main():
    # Load configuration
    config = load_config()
    data_path = config.get("final_merged_data_path")
    logger.info(f"Loaded data from {data_path}")
    if not data_path or not os.path.exists(data_path):
        logger.error(f"Data path {data_path} does not exist in config.")
        return

    spark = create_spark_session(name="TrainTestSplit")
    df = spark.read.parquet(data_path)
    logger.info(f"Loaded data with {df.count()} records and {len(df.columns)} columns.")
    count_distinct_ip = df.select("ip_address").distinct().count()
    print(f"Số lượng IP duy nhất: {count_distinct_ip}")

    train_df = df.filter((F.col("window_5min_end") <= F.lit(SPLIT_POINT)) & (F.col("window_5min_end")>= F.lit(START_POINT)))
    test_df = df.filter((F.col("window_5min_end") > F.lit(SPLIT_POINT)) & (F.col("window_5min_end") <= F.lit(COT_POINT)))

    # statistics for train and test data
    train_df = train_df.persist()
    test_df = test_df.persist()

    # Number of records
    train_count = train_df.count()
    test_count = test_df.count()
    logger.info(f"Train set records: {train_count:,.0f}")
    logger.info(f"Test set records: {test_count:,.0f}")
    
    # Timerange
    train_df.select(F.min("window_5min_end").alias("min_TRAIN_time"), F.max("window_5min_end").alias("max_TRAIN_time")).show(truncate=False)
    test_df.select(F.min("window_5min_end").alias("min_TEST_time"), F.max("window_5min_end").alias("max_TEST_time")).show(truncate=False)

    # label_rate
    train_anom_count = train_df.filter(F.col("label") == 1).count()
    test_anom_count = test_df.filter(F.col("label") == 1).count()
    train_label_rate = train_anom_count / train_count if train_count > 0 else 0
    test_label_rate = test_anom_count / test_count if test_count > 0 else 0
    logger.info(f"Train set anomaly count: {train_anom_count:,.0f} ({train_label_rate:.4%})")
    logger.info(f"Test set anomaly count: {test_anom_count:,.0f} ({test_label_rate:.4%})")

    # Save split data
    train_output_path = os.path.join(SPLIT_DIR, "train_data_with_linecard_ext_MarApr")
    test_output_path = os.path.join(SPLIT_DIR, "test_data_with_linecard_ext_MarApr")
    train_df.write.mode("overwrite").parquet(train_output_path)
    test_df.write.mode("overwrite").parquet(test_output_path)

    logger.info(f"Train and Test data saved to {SPLIT_DIR}")
    config["train_data_path"] = os.path.abspath(train_output_path)
    config["test_data_path"] = os.path.abspath(test_output_path)
    save_config(config)
    logger.info(f"Updated config.yaml with train and test data paths.")

if __name__ == "__main__":
    main()
