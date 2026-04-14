import os
import sys
import logging
from datetime import datetime
import pytz
vntz = pytz.timezone('Asia/Ho_Chi_Minh')
import warnings
warnings.filterwarnings("ignore")
import glob
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, LongType

ROOT_DIR = r""
sys.path.insert(0,ROOT_DIR)
from src.utils.helper_func import load_config, save_config, create_spark_session, export_data, flatten_value_column, load_and_normalize_parquet

LABEL_OUTPUT = os.path.join(ROOT_DIR, "data", "raw", "down_ip_Oct_upto7th")
os.makedirs(LABEL_OUTPUT, exist_ok=True)

MAPPED_NOC_DIR = r""
SNMP_DIR = r""

LOG_DIR = os.path.join(ROOT_DIR, "artifacts", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(os.path.join(LOG_DIR, f"label_generator_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}.log"))])
logger = logging.getLogger(__name__)

# read only data from A to B

day_range = range(,)
month_range = range(,)

def load_snmp_data(spark):

    # bộ ip noc_mapped
    mapped_noc_files = glob.glob(MAPPED_NOC_DIR, recursive=True)
    logger.info(f"Found {len(mapped_noc_files)} mapped NOC files.")
    df_mapped_noc = spark.read.parquet(*mapped_noc_files)
    spark.conf.set("spark.sql.caseSensitive", "true")
    unique_ip_set = df_mapped_noc.select(F.collect_set("ip_address")).collect()[0][0]

    value_schema = StructType([
            StructField("cpu",          StringType(), True),
            StructField("heap",         StringType(), True),
            StructField("state",        StringType(), True),
            StructField("temp",         StringType(), True),
        ])
    
    schema = StructType([
        # StructField("job_id",           StringType(), True),
        # StructField("department",       StringType(), True),
        # StructField("protocol",         StringType(), True),
        StructField("ip-address",       StringType(), True),
        # StructField("device_type",      StringType(), True),
        StructField("component_name",   StringType(), True),
        # StructField("component_index",  StringType(), True),
        StructField("value", value_schema, True),
        StructField("time",             StringType(), True),
        StructField("service_name",     StringType(), True),
        # StructField("is_retry",         BooleanType(), True),
    ])

    ## read only Apr and May
    files_to_read = [
        path for d in day_range
        for m in month_range
        for path in glob.glob(f"{SNMP_DIR}/month={m:02d}/day={d:02d}/*.parquet", recursive=True)
    ]
    # df_comp = spark.read.schema(schema).parquet(*files_to_read)
    logger.info(f"Falling back to reading files one by one.")
    df_comp = None
    for item in files_to_read:
        df = load_and_normalize_parquet(spark, item, schema, value_schema)
        if df_comp is None:
            df_comp = df
        else:
            df_comp = df_comp.union(df)

    logger.info(f"Read {len(files_to_read)} SNMP files from {SNMP_DIR}")
    df_comp_lim = df_comp.filter(F.col("service_name")=="check_juniper_component") 

    ## lọc comp fpc|re và ip
    component_sc_filter = F.col("component_name").rlike(r"(?i)fpc|routing engine")
    df_comp_lim = df_comp_lim.filter(
                                (F.col("ip-address").isin(list(set(unique_ip_set)))) & 
                                (component_sc_filter))
    
    ## flatten value col
    df_comp_lim = flatten_value_column(df_comp_lim)
    df_comp_lim = df_comp_lim.withColumn("time", F.to_timestamp(F.col("time")))

    DOWN_cond = ~F.col("state").try_cast(LongType()).isin([2,7])
    df_label_down = df_comp_lim.filter(DOWN_cond)\
                        .withColumn("agg_status", F.lit("DOWN"))\
                        .withColumnRenamed("ip-address", "ip_address")\
                        .select("ip_address", "time", "agg_status")\
                        .dropDuplicates()

    export_data(df_label_down, LABEL_OUTPUT)
    logger.info(f"Label data saved to {LABEL_OUTPUT}")

    df_label_down.select(F.min("time"), F.max("time")).show(truncate=False)

if __name__ == "__main__":
    spark = create_spark_session("LabelGenerator")
    load_snmp_data(spark)
    spark.stop()
    logger.info("Label generation completed.")