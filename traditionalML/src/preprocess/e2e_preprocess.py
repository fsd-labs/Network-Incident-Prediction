import pandas as pd
import numpy as np
import logging
from pyspark.sql import SparkSession, DataFrame, Window
import pyspark.sql.functions as F
import glob, re, os, sys
from pathlib import Path
import time
from functools import wraps
import warnings
import ipaddress
from datetime import datetime, timedelta
from pyspark.sql.types import (
    StructType, StructField,
    StringType, BooleanType, LongType, StructType, FloatType, DoubleType, ArrayType
)
from functools import reduce

warnings.filterwarnings("ignore")
import pytz
vntz = pytz.timezone('Asia/Ho_Chi_Minh')

MAPPED_NOC_DIR = r""
SNMP_DIR = r""
SYSLOG_DIR = r""
LABEL_DIR = r""


# add
BRAS_DIR = r""
ROOT_DIR = r""
sys.path.insert(0,ROOT_DIR)
from src.utils.helper_func import load_config, save_config, create_spark_session, export_data, flatten_value_column, load_and_normalize_parquet_v2, find_value_type_files

# define OUTPUT_DIR
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_DIR = os.path.join(ROOT_DIR, "artifacts", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

COT_TIME = ""
UPPER_CUT = "

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(LOG_DIR, f"e2e_pre_{datetime.now(vntz).strftime('%Y-%m-%d_%H-%M')}_extra.log"))]
)

logger = logging.getLogger(__name__)

def create_n_min_window_bucket(df: DataFrame, time_col="time", n=5):
    df = df.withColumn("ts_midnight", F.date_trunc("day", F.col(time_col)))\
                        .withColumn("bucket_start_min", F.floor((F.hour(time_col) * 60 + F.minute(time_col)) / n) * n)\
                        .withColumn(f"window_{n}min_end", F.from_unixtime(
                            F.unix_timestamp("ts_midnight") +
                            (F.col("bucket_start_min") + n) * 60).cast("timestamp"))\
                        .drop("ts_midnight", "bucket_start_min")
    return df

def generate_rolling_features(df: DataFrame, metrics: list, time_step: list, time_col= "t_secs"):
    new_features = []
    for metric in metrics:
        for ws in time_step:
            new_features.extend([
                F.avg(F.col(metric)).over(Window.partitionBy("ip-address").orderBy(time_col).rangeBetween(-ws, 0)).alias(f"{metric}_avg_{ws}"),
                F.stddev(F.col(metric)).over(Window.partitionBy("ip-address").orderBy(time_col).rangeBetween(-ws, 0)).alias(f"{metric}_std_{ws}"),
                # pct change between the avg of the last two windows
                F.try_divide((F.avg(F.col(metric)).over(Window.partitionBy("ip-address").orderBy(time_col).rangeBetween(-ws, 0)) - 
                 F.avg(F.col(metric)).over(Window.partitionBy("ip-address").orderBy(time_col).rangeBetween(-2*ws, -ws))),
                 F.avg(F.col(metric)).over(Window.partitionBy("ip-address").orderBy(time_col).rangeBetween(-2*ws, -ws))
                ).alias(f"{metric}_pct_change_{ws}")
            ])
    
    return df.select("*", *new_features)

# calculate ratio countlog with different delta
def calculate_logcount_ratio(df: DataFrame, time_col: str, delta: list):
    new_feature_list = []
    for d in delta:
        w1 = Window.partitionBy("host").orderBy(time_col).rangeBetween(-d, 0)
        w2 = Window.partitionBy("host").orderBy(time_col).rangeBetween(-2*d, -d)
        new_feature_list.extend([
            F.try_divide(F.count("*").over(w1), F.count("*").over(w2)).alias(f"ratio_countlog_delta_{d/60:.0f}min")
        ])
    return df.select("*", *new_feature_list)

# calculate priority score (facility + severity * 8)
def calculate_priority_score(df: DataFrame, pri_col = "priority_score", time_col = "timestamp_sec", windows = [900]):
    new_feats = []
    for w in windows:
        # ratio of mean priority score in the last w seconds to the previous w seconds
        new_feats.extend([
            # avg priority score in the last w seconds
            F.avg(F.col(pri_col)).over(Window.partitionBy("host").orderBy(time_col).rangeBetween(-w, 0)).alias(f"avg_pri_score_prev_{w/60:.0f}m"),
            # 15 min vs 30 min
            F.try_divide(F.col(f"avg_pri_score_prev_{w/60:.0f}m"),
             F.avg(F.col(pri_col)).over(Window.partitionBy("host").orderBy(time_col).rangeBetween(-2*w, 0))).alias(f"ratio_pri_score_prev_{w/60:.0f}m_vs_{w*2/60:.0f}m"),
            # 15min vs 60 min
            F.try_divide(F.col(f"avg_pri_score_prev_{w/60:.0f}m"),
                F.avg(F.col(pri_col)).over(Window.partitionBy("host").orderBy(time_col).rangeBetween(-4*w, 0))).alias(f"ratio_pri_score_prev_{w/60:.0f}m_vs_{w*4/60:.0f}m")
        ])
    
    df = df.select("*", *new_feats)\
            .drop(pri_col)
    return df


def agg_n_min_window_bucket(df:DataFrame):
    df = create_n_min_window_bucket(df, time_col="time", n=5)
    row_struct = F.struct(*[
            F.col(c) for c in df.columns 
            if c not in ("ip-address","window_5min_end", "time") #, "t_secs")
        ]).alias("last_row")

    df = (
            df
            .groupBy("ip-address", "window_5min_end")
            .agg(
                F.max_by(row_struct, F.col("time")).alias("last_row")
            )
            # unpack it back into columns
            .select(
                F.col("ip-address").alias("ip_address"),
                "window_5min_end",
                "last_row.*"
            )
        )
    return df


def main(pre_snmp, pre_sys, merge):
    spark = create_spark_session(name="E2E Feature Engineering Pipeline")
    spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    # spark.sparkContext.setLogLevel("INFO")
    spark.conf.set("spark.sql.caseSensitive", "true")
    
    #bộ ip noc_mapped
    mapped_noc_files = glob.glob(MAPPED_NOC_DIR, recursive=True)
    df_mapped_noc = spark.read.parquet(*mapped_noc_files)
    unique_ip_set = df_mapped_noc.select(F.collect_set("ip_address")).collect()[0][0]

    # bộ ip bras
    bras_files = glob.glob(BRAS_DIR, recursive=True)
    bras_schema = StructType([
        StructField("local_device_ip", StringType(), True),
        StructField("remote_device_ip", StringType(), True),
    ])
    bras_df = spark.read.schema(bras_schema).parquet(*bras_files)
    bras_ctr_ip_set = bras_df.select(F.collect_set("local_device_ip")).collect()[0][0]
    bras_pe_ip_set = bras_df.select(F.collect_set("remote_device_ip")).collect()[0][0]

    # ip down extra
    down_noc_files = glob.glob(LABEL_DIR, recursive=True)
    snmp_down_ip_time = spark.read.parquet(*down_noc_files)
    snmp_down_ip_time =  snmp_down_ip_time.select(F.col("ip-address").alias("ip_address"), "time").dropDuplicates()
    uppercut_mod = (pd.to_datetime(UPPER_CUT) + pd.Timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Filtering label sets with time <= {uppercut_mod}")
    snmp_down_ip_time = snmp_down_ip_time.withColumn("time", F.to_timestamp(F.col("time")))\
                            .filter(F.col("time") <= pd.to_datetime(uppercut_mod))
    snmp_down_ip_time = snmp_down_ip_time.persist()
    logger.info(f"Total {snmp_down_ip_time.count()} down IP-Time records loaded from LABEL_DIR, min time: {snmp_down_ip_time.agg(F.min('time')).collect()[0][0]}, max time: {snmp_down_ip_time.agg(F.max('time')).collect()[0][0]}")

    list_ip_down = snmp_down_ip_time.select(F.collect_set("ip_address")).collect()[0][0]
    list_all_ip = list(set(bras_ctr_ip_set + bras_pe_ip_set + unique_ip_set))
    list_host = list(set(list_ip_down) & set(list_all_ip))
    
    if "118.70.0.85" in list_host:
        list_host.remove("118.70.0.85")

    print(f"List {len(list_host)} host is: \n {list_host}")
    #=======================SNMP DATA=======================
    if not pre_snmp:
        logger.info("Skipping SNMP data preprocessing as per user request.")
    else:
        logger.info("STARTING SNMP DATA PREPROCESSING...")

        value_schema = StructType([
                StructField("cpu",          StringType(), True),
                StructField("heap",         StringType(), True),
                StructField("state",        StringType(), True),
                StructField("temp",         StringType(), True),
                StructField("jnx_subscriber_active_count", StringType(), True),
                StructField("jnx_subscriber_total_count", StringType(), True),
                StructField("if_out_octets", StringType(), True),
                StructField("if_in_octets", StringType(), True),
            ])

        schema = StructType([
            StructField("job_id",           StringType(), True),
            StructField("department",       StringType(), True),
            StructField("protocol",         StringType(), True),
            StructField("ip-address",       StringType(), True),
            StructField("device_type",      StringType(), True),
            StructField("component_name",   StringType(), True),
            StructField("if_name",          StringType(), True),
            StructField("component_index",  StringType(), True),
            StructField("value",            value_schema, True),
            StructField("time",             StringType(), True),
            StructField("service_name",     StringType(), True),
            StructField("is_retry",         BooleanType(),True),
        ])

        files_to_read = []
        for m in [11,12]:
            if m in [11]:
                days = list(range(21, 32))
                files_to_read.extend([
                    path for d in days
                    for path in glob.glob(f"{SNMP_DIR}/month={m:02d}/day={d:02d}/*.parquet", recursive=True)
                ])
                logger.info(f"Added {len(files_to_read)} files for month={m:02d}")
 
        string_value_files = []
        struct_value_files = []
        error_files = []

        string_f, struct_f, error_f = find_value_type_files(spark, files_to_read)
        string_value_files.extend(string_f)
        struct_value_files.extend(struct_f)
        error_files.extend(error_f)

        df1 = load_and_normalize_parquet_v2(spark, string_value_files, schema, value_schema)
        df2 = load_and_normalize_parquet_v2(spark, struct_value_files, schema, value_schema)
        if df1 is None: 
            df_snmp = df2
        elif df2 is None: 
            df_snmp = df1 
        else: 
            df_snmp = df1.unionByName(df2, allowMissingColumns=True)
        logger.info(f"Read {len(files_to_read)} SNMP files from {SNMP_DIR}")

        df_snmp = df_snmp.filter(F.col("ip-address").isin(list_host))

        # ============ SNMP COMPONENT ============ 
        df_comp_lim = df_snmp.filter(F.col("service_name")=="check_juniper_component") 
        ## lọc comp fpc|re và ip
        component_sc_filter = F.col("component_name").rlike(r"(?i)fpc|routing engine")
        df_comp_lim = df_comp_lim.filter(component_sc_filter)
        
        ## flatten value col
        df_comp_lim = flatten_value_column(df_comp_lim)
        df_comp_lim = df_comp_lim.withColumn("time", F.to_timestamp(F.col("time"))).dropDuplicates()

        ## regex component name
        df_comp_lim = df_comp_lim.withColumn("component_name", F.when(F.col("component_name") == "Routing Engine", "Routing Engine 0").otherwise(F.col("component_name")))\
                                .withColumn("fpc_number", F.regexp_extract(F.col("component_name"), r"@ (\d+)", 1))\
                                .withColumn("regex_component_name", F.when(
                                    F.col("component_name").rlike(r"(?i)fpc"),
                                    F.concat(F.lit("FPC "), F.col("fpc_number"))
                                ).otherwise(F.col("component_name")))\
                                .drop("fpc_number")

        ## pivot cpu, heap, state, temp theo regex_component_name
        df_comp_lim = df_comp_lim.repartition("ip-address", "time")
        df_comp_lim = df_comp_lim.groupBy("ip-address", "time")\
            .pivot("regex_component_name", df_comp_lim.select("regex_component_name").distinct().rdd.flatMap(lambda x: x).collect()) \
            .agg(
                F.first("cpu").alias("cpu"),
                F.first("heap").alias("heap"),  
                F.first("state").alias("state"),
                F.first("temp").alias("temp")
            )

        conv_cols = [col for col in df_comp_lim.columns if bool(re.search(r"(cpu|heap|temp)", col))]
        state_cols = [col for col in df_comp_lim.columns if bool(re.search(r"state", col))]
        ## convert cpu, heap, temp to float
        df_comp_lim = df_comp_lim.select("ip-address", "time", *state_cols, *[F.col(col).try_cast("float").alias(col) for col in conv_cols])\
                                .withColumn("t_secs", F.unix_timestamp(F.col("time")))

        ## generate rolling features
        ws = [900, 1800, 3600]
        df_comp_lim = generate_rolling_features(df_comp_lim, conv_cols, ws)
        ## states the repeat the most in the last N window and the 2nd last N window
        mode_state_feats = []
        for col in state_cols:
            mode_state_feats.extend([
                F.mode(F.col(col)).over(Window.partitionBy("ip-address").orderBy("t_secs").rangeBetween(-1800, 0)).alias(f"{col}_mode_last_30min"),
                F.mode(F.col(col)).over(Window.partitionBy("ip-address").orderBy("t_secs").rangeBetween(-3600, -1800)).alias(f"{col}_mode_2nd_last_30min")
            ])

        df_comp_lim = df_comp_lim.select("*", *mode_state_feats)
        df_comp_lim = agg_n_min_window_bucket(df_comp_lim)
        
        logger.info("Rolling features generated for CPU, Heap, and Temp.")
        ## ============ END SNMP COMPONENT ============  

        # ============ SNMP Traffic ============ 
        df_traffic_lim = df_snmp.filter(F.col("service_name")=="check_juniper_interface_statictis") 
        df_traffic_lim = df_traffic_lim.filter(F.col("if_name").startswith("et-")) 
        ## flatten value col
        df_traffic_lim = df_traffic_lim.select("*", F.col("value.if_out_octets").alias("if_out_octets"),
                                                    F.col("value.if_in_octets").alias("if_in_octets")) \
                                                        .drop("value")
        df_traffic_lim = df_traffic_lim.withColumn("time", F.to_timestamp(F.col("time"))).dropDuplicates()
        df_traffic_lim = df_traffic_lim.repartition("ip-address", "time")

        conv_cols = ["if_out_octets", "if_in_octets"]
        ## convert cpu, heap, temp to float
        df_traffic_lim = df_traffic_lim.select("ip-address", "time", *[F.col(col).try_cast("float").alias(col) for col in conv_cols])\
                                .withColumn("t_secs", F.unix_timestamp(F.col("time")))

        ## generate rolling features
        ws = [900, 1800]
        df_traffic_lim = generate_rolling_features(df_traffic_lim, conv_cols, ws)
        df_traffic_lim = agg_n_min_window_bucket(df_traffic_lim)
        logger.info("Rolling features generated for if_out_octets, if_out_octets")
        ## ============ END SNMP Traffic ============  

        # ============ SNMP Subscriber ============ 
        df_subscriber_lim = df_snmp.filter(F.col("service_name")=="check_juniper_subscribers") 
        df_subscriber_lim.printSchema()
        ## flatten value col
        df_subscriber_lim = df_subscriber_lim.select("*",  
                                                F.col("value.jnx_subscriber_active_count").alias("jnx_subscriber_active_count"),
                                                F.col("value.jnx_subscriber_total_count").alias("jnx_subscriber_total_count")) \
                                        .drop("value")
        df_subscriber_lim = df_subscriber_lim.withColumn("time", F.to_timestamp(F.col("time"))).dropDuplicates()
        df_subscriber_lim = df_subscriber_lim.repartition("ip-address", "time")

        conv_cols = ["jnx_subscriber_active_count", "jnx_subscriber_total_count"]
        ## convert cpu, heap, temp to float
        df_subscriber_lim = df_subscriber_lim.select("ip-address", "time", *[F.col(col).try_cast("float").alias(col) for col in conv_cols])\
                                .withColumn("t_secs", F.unix_timestamp(F.col("time")))

        ## generate rolling features
        ws = [900, 1800, 3600]
        df_subscriber_lim = generate_rolling_features(df_subscriber_lim, conv_cols, ws)
        df_subscriber_lim = agg_n_min_window_bucket(df_subscriber_lim)
        logger.info("Rolling features generated for jnx_subscriber_active_count, jnx_subscriber_total_count")
        
        ## ============ END SNMP Traffic ============  

        ## union 3 df 
        dfs = [df_comp_lim, df_subscriber_lim, df_traffic_lim]
        df_union = reduce(
            lambda df_left, df_right: df_left.unionByName(df_right, allowMissingColumns=True),
            dfs
        )
        data_cols = [c for c in df_union.columns if c not in ["ip_address", "window_5min_end", "t_secs"]]
        row_struct = F.struct(*[F.col(c).alias(c) for c in data_cols])
        df_union = df_union.select(
            "ip_address",
            "window_5min_end",
            row_struct.alias("row_struct")
        )

        df_union = (    
            df_union
            .groupBy("ip_address", "window_5min_end")
            .agg(F.collect_list("row_struct").alias("rows"))
        )

        df_union = df_union.select(
            F.col("ip_address"),
            "window_5min_end",
            *[
                F.expr(
                    f"get(filter(rows, x -> x.`{c}` IS NOT NULL), 0).`{c}`"
                ).alias(c)
                for c in data_cols
            ]
        )

        logger.info("SNMP DATA PREPROCESSING COMPLETED.")
        df_union.printSchema()

        df_union = df_union.filter(F.col("window_5min_end") >= COT_TIME)\
                                 .repartition("ip_address", "window_5min_end")
        df_union = df_union.persist()

        print("FINAL TIME RANGE FOR SNMP DATA:")
        df_union.select(F.min("window_5min_end").alias("min_snmp_ts"), F.max("window_5min_end").alias("max_snmp_ts")).show(truncate=False)
        
        logger.info(f"EXPORTING SNMP DATA TO {OUTPUT_DIR}")
        export_data(df_union, os.path.join(OUTPUT_DIR, "snmp_data_preprocessed"), mode="overwrite")

        existing_config = load_config()
        if existing_config is None:
            existing_config = {}
        existing_config["snmp_data_path"] = os.path.abspath(os.path.join(OUTPUT_DIR, "snmp_data_preprocessed"))
        save_config(existing_config)
        logger.info(f"Updated config.yaml with SNMP data path.")

        spark.catalog.clearCache()
        df_union.unpersist()

    #=======================SYSLOG DATA (combined_message, count & priority score)=======================
    if not pre_sys:
        logger.info(f"Skipping SYSLOG data preprocessing as per user request.")
    else:
        logger.info(f"START PREPROCESSING SYSLOG DATA...") 

        syslog_files = []
        for m in [11,12]:
            if m in [11]:
                days = list(range(21, 32))
                syslog_files.extend([
                    path for d in days
                    for path in glob.glob(f"{SYSLOG_DIR}/2026-{m:02d}-{d:02d}/*.parquet", recursive=True)
                ])
                logger.info(f"Added {len(syslog_files)} files for month={m:02d}")

        syslog_schema = StructType([
            StructField("syslog_pri", StringType(), True),
            StructField("message", StringType(), True),
            StructField("host", StringType(), True),
            StructField("@timestamp", StringType(), True),
        ])
        sys_df = spark.read.schema(syslog_schema).parquet(*syslog_files)
        logger.info(f"Read {len(syslog_files)} syslog files from {SYSLOG_DIR}")

        sys_df = sys_df.withColumn("syslog_pri", F.col("syslog_pri").try_cast("double"))
        # list host not running done
        sys_df = sys_df.filter(F.col("host").isin(list_host))\
                .withColumn("timestamp", F.to_timestamp(F.col("@timestamp")))\
                .drop("@timestamp") \
                .dropDuplicates()
        
        logger.info("============================================================================")
        ## REGEX TO EXTRACT FROM RAW MESSAGE
        raw = F.col("message")                      # original text
        pats = [
            # 1) process[pid] + %FAC-SEV-TAG:
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\[(\d+)\]:\s+%"
            r"([A-Z]+)-(\d+)-([A-Z0-9_]+):\s+(.+)$",
            dict(ts=1,h=2,p=3,pid=4,fac=5,sev=6,tag=7,msg=8)),
            # 2) process[pid] + %FAC-SEV:    (no tag)
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\[(\d+)\]:\s+%"
            r"([A-Z]+)-(\d+):\s+(.+)$",
            dict(ts=1,h=2,p=3,pid=4,fac=5,sev=6,msg=7)),
            # 3) process + %FAC-SEV-TAG:  (extended chars in TAG)
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+%"
            r"([A-Z]+)-(\d+)-([A-Za-z0-9_\-]+):\s+(.+)$",
            dict(ts=1,h=2,p=3,fac=4,sev=5,tag=6,msg=7)),
            # 4) process + %FAC-SEV:  (no tag, optional spaces)
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s*%"
            r"([A-Z]+)-(\d+):\s+(.+)$",
            dict(ts=1,h=2,p=3,fac=4,sev=5,msg=6)),
            # 5) host-only  + %FAC-SEV:
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+:\s+%"
            r"([A-Z]+)-(\d+):\s+(.+)$",
            dict(ts=1,h=2,fac=3,sev=4,msg=5)),
            # 6) process + TAG:   (no % block)
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+"
            r"([A-Z0-9_]+):\s+(.+)$",
            dict(ts=1,h=2,p=3,tag=4,msg=5)),
            # 7) “|<pri>” style  %FAC-SEV-TAG:
            (r"^\|?<\d+>\d+:\s+(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}):\s+%"
            r"([A-Z_]+)-(\d+)-([A-Z0-9_]+):\s+(.+)$",
            dict(ts=1,fac=2,sev=3,tag=4,msg=5)),
            # 8) “last message repeated N times”
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+"
            r"(last message repeated \d+ times)$",
            dict(ts=1,h=2,msg=3)),
            # 9) process + free text   (no %, no tag)
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(.+)$",
            dict(ts=1,h=2,p=3,msg=4)),
            # 10) process[pid]: TAG: text   (no %)
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\[(\d+)\]:\s+"
            r"([A-Za-z0-9_:\-]+):\s+(.+)$",
            dict(ts=1,h=2,p=3,pid=4,tag=5,msg=6)),
            # 11) process  TAG: text  (space before TAG, no %)
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\s+"
            r"([A-Za-z0-9_\-]+):\s+(.+)$",
            dict(ts=1,h=2,p=3,tag=4,msg=5)),
            # 12) host process raw text  (no colon)
            (r"^<\d+>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\s+(.+)$",
            dict(ts=1,h=2,p=3,msg=4)),
        ]
        def build_struct(rx, m):
            def g(k):
                return F.regexp_extract(raw, rx, m[k]) if k in m else F.lit(None).cast("string")
            return F.struct(
                g("ts").alias("timestamp"),
                g("msg").alias("message")
            )
        # pick the first pattern that matches
        parsed_candidates = [F.when(raw.rlike(rx), build_struct(rx, mp)) for rx, mp in pats]
        sys_df = sys_df.withColumn(
            "parsed",
            reduce(lambda a, b: F.coalesce(a, b), parsed_candidates)
        )
        # Message Clean-up
        repls = [
            (r"[()\[\],\->=%*\\><!'\"]+", " "),       # punctuation
            (r" - |--",               " "),           # two-char dashes
            (r"\b\d{1,3}(?:\.\d{1,3}){3}\b",  ""),    # IP
            (r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b", ""),  # MAC
            (r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b", ""),   # yyyy-mm-dd hh:mm:ss
            (r"\b\d{2}:\d{2}:\d{2}\b", ""),           # hh:mm:ss
            (r"\b(?!100\b)\d{3,}\b", ""),             # numbers > 100 (keep ‘100’)
            (r"[:.\-]+", " "),                        # split compound tokens
            (r"\s+", " ")                             # collapse whitespace
        ]
        clean_msg = reduce(lambda c, r: F.regexp_replace(c, r[0], r[1]),
                        repls, F.col("parsed.message"))
        msg_final = F.regexp_replace(clean_msg,   r"^\[\w+\]\s*", "")
        sys_df = sys_df.withColumn(
            "parsed",
            F.when(F.col("parsed").isNotNull(),
                F.struct(
                    "parsed.timestamp",
                    F.trim(msg_final).alias("message")
                ))
        )
        sys_df = sys_df.select(
            F.col("parsed.timestamp").alias("device_time"),
            F.col("syslog_pri").alias("priority_score"),
            F.lower(F.col("parsed.message")).alias("message"),
            F.col("host").alias("host"),
            F.col("timestamp").alias("timestamp"),
        )
        sys_df = sys_df.filter(F.col("device_time").isNotNull())\
                        .drop("device_time")
        
        sys_df = sys_df.withColumn("timestamp_sec", F.unix_timestamp(F.col("timestamp")))
        sys_df = sys_df.repartition("host")
        
        ## calculate ratio countlog
        deltas = [1800, 2700]
        sys_df = calculate_logcount_ratio(sys_df, "timestamp_sec", deltas)
        logger.info(f"Log count ratio features generated for deltas: {deltas}")
        ## calculate priority score:
        ws = [900]
        sys_df = calculate_priority_score(sys_df, pri_col="priority_score", time_col = "timestamp_sec", windows = ws)
        logger.info(f"Priority score features generated for syslog data with windows: {ws}")
        ## create 5 min window bucket:
        sys_df = create_n_min_window_bucket(sys_df, time_col="timestamp", n=5)
        ## (Refactored) sort lại đúng thứ tự log trong từng nhóm message
        agg_arr = F.collect_list(
                F.struct(
                    F.col("timestamp_sec").alias("ts"),
                    F.col("message").alias("msg")
                )
            ).alias("arr")
        
        row_struct = F.struct(*[F.col(c) for c in sys_df.columns if c not in ["host", "timestamp_sec", "timestamp", "window_5min_end", "message"]]).alias("last_row")
        sys_df = sys_df.groupBy("host", "window_5min_end")\
                        .agg(
                            agg_arr,
                            F.max_by(row_struct, F.col("timestamp")).alias("last_row_struct")
                            )
        sys_df = sys_df.withColumn("combined_message",
                                    F.concat_ws(
                                        " ",
                                        F.transform(F.array_sort(F.col("arr")), lambda x: x.msg)
                                    ))\
                        .drop("arr")
        sys_df = sys_df.select(
            "host", 
            "window_5min_end", 
            "last_row_struct.*", 
            F.lower("combined_message").alias("combined_message")
        ).filter(F.col("window_5min_end") >= COT_TIME)
        logger.info(f"SYSLOG DATA PREPROCESSING COMPLETED.")
        sys_df.printSchema()

        # sys_df = sys_df.persist()
        # print("FINAL TIME RANGE FOR SYSLOG DATA:")
        # sys_df.select(F.min("window_5min_end").alias("min_syslog_ts"), F.max("window_5min_end").alias("max_syslog_ts")).show(truncate=False)

        logger.info(f"EXPORTING SYSLOG DATA TO {OUTPUT_DIR}")
        export_data(sys_df, os.path.join(OUTPUT_DIR, "syslog_data_preprocessed"), mode="overwrite")

        existing_config = load_config()
        if existing_config is None:
            existing_config = {}
        existing_config["syslog_data_path"] = os.path.abspath(os.path.join(OUTPUT_DIR, "syslog_data_preprocessed"))
        save_config(existing_config)
        logger.info(f"Updated config.yaml with SYSLOG data path.")

        sys_df.unpersist()
        spark.catalog.clearCache()

    #=======================COMBINE SNMP, SYSLOG DATA AND LABEL SET=======================
    if not merge:
        logger.info(f"Skipping final Data Merging as per user request")
    else:
        logger.info(f"START MERGING SNMP DATA, SYSLOG DATA AND LABEL SET")

        df_comp_lim = spark.read.parquet(os.path.join(OUTPUT_DIR, "snmp_data_preprocessed"))
        df_comp_lim = df_comp_lim.filter(F.col("window_5min_end") <= UPPER_CUT)

        sys_df = spark.read.parquet(os.path.join(OUTPUT_DIR, "syslog_data_preprocessed"))
        sys_df = sys_df.filter(F.col("window_5min_end") <= UPPER_CUT)
        logger.info(f"LOADED SNMP and SYSLOG PREPROCESSED DATA FROM {OUTPUT_DIR}")

        logger.info("===============================================================================")
        logger.info("Checking time range of SNMP data...")
        df_comp_lim.select(F.min("window_5min_end"), F.max("window_5min_end")).show()
        logger.info("Checking time range of SYSLOG data...")
        sys_df.select(F.min("window_5min_end"), F.max("window_5min_end")).show()
        logger.info("===============================================================================")

        # merge and label
        ## syslog and label
        sys_df = sys_df.withColumn("time_upper", F.expr("window_5min_end + INTERVAL 20 minute"))\
                        .repartition("host", "window_5min_end")
        snmp_down_ip_time = snmp_down_ip_time.repartition("ip_address", "time")

        sys_gr = sys_df.alias("sys_gr")
        snmp_lab = snmp_down_ip_time.alias("snmp_lab")

        merged = sys_gr.join(F.broadcast(snmp_lab), 
                             F.expr("""(sys_gr.host = snmp_lab.ip_address)
                                    AND (snmp_lab.time BETWEEN sys_gr.window_5min_end AND sys_gr.time_upper)"""),
                             how = "left")
    
        merged = merged.withColumn("timediff", F.unix_timestamp("time") - F.unix_timestamp("window_5min_end"))\
                        .filter(~(
                            (F.col("ip_address").isNotNull()) &
                            (F.col("timediff") < 60*5)
                        ))\
                        .withColumn("label", F.when(F.col("ip_address").isNotNull(), F.lit(1)).otherwise(F.lit(0)))\
                        .drop("timediff", "time_upper", "time", "agg_status", "ip_address")\
                        .withColumnRenamed("host", "ip_address")\
                        .dropDuplicates()
        
        ## snmp to the merged
        df_comp_lim = df_comp_lim.repartition("ip_address", "window_5min_end")
        merged = merged.join(df_comp_lim, on = ["ip_address", "window_5min_end"], how = "left")
        logger.info("Merged SNMP and SYSLOG data with label set.")

        logger.info(f"FINAL SCHEMA AS MODEL INPUT")
        merged.printSchema()

        merged = merged.persist()
        logger.info("===============================================================================")
        logger.info("CHECKING TIMERANGE OF THE FINAL MERGED DATA...")
        merged.select(F.min("window_5min_end").alias("min_merged_ts"), F.max("window_5min_end").alias("max_merged_ts")).show(truncate=False)
        logger.info("===============================================================================")

        logger.info("Exporting final merged data to Parquet format...")
        export_data(merged, os.path.join(OUTPUT_DIR, "final_merged_data_latest"), mode="overwrite")
        logger.info(f"FINAL - TOTAL NUMBER OF FEATURES: {len(merged.columns)}")

        existing_config = load_config()
        if existing_config is None:
            existing_config = {}
        existing_config["final_merged_data_path"] = os.path.abspath(os.path.join(OUTPUT_DIR, "final_merged_data_latest"))
        save_config(existing_config)
        logger.info(f"Updated config.yaml with final merged data path.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-snmp", "--preprocess_snmp", action="store_true")
    p.add_argument("-sys", "--preprocess_syslog", action="store_true")
    p.add_argument("-merge", "--merge_data", action="store_true")
    args = p.parse_args()
    main(pre_snmp = args.preprocess_snmp, pre_sys = args.preprocess_syslog, merge = args.merge_data)