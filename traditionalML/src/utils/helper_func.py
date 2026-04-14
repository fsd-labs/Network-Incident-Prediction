import yaml
import os, sys
import warnings
warnings.filterwarnings("ignore")
import pyspark
from pyspark.sql import SparkSession
import logging
import time
from datetime import datetime
from pyspark.sql import DataFrame, functions as F
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay, confusion_matrix
import glob
import pandas as pd
import warnings
import re
import json
warnings.filterwarnings("ignore")
from pathlib import Path
from pyspark.sql.types import (
    StructType, StructField,
    StringType, BooleanType, LongType, StructType, FloatType, DoubleType, ArrayType
)

logger = logging.getLogger(__name__)

CONFIG_PATH = r"/home/ubuntu/AnomalyDetection/dev_refactor/src/config/config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            raise e

def save_config(config, config_path=CONFIG_PATH):
    with open(config_path, 'w') as file:
        try:
            yaml.safe_dump(config, file)
        except yaml.YAMLError as e:
            logger.error(f"Error saving configuration file: {str(e)}")
            raise e

def load_hyperparameters(config_path):
    """Include: 
    - common search space for LightGBM
    - Specific best params for each model version (model_v1: ...)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(path_pattern: str, filter_by_incidents: bool = True) -> pd.DataFrame:
    logger.info(f"Loading data from: {path_pattern}")
    desired_pat = "*.parquet"
    files = glob.glob(os.path.join(path_pattern, "**", desired_pat), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {path_pattern}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    # df = df.iloc[:1000]
    logger.info(f"Loaded {len(files)} files; shape={df.shape}")
    return df

def to_builtin(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (pd.Timedelta, np.timedelta64)):
        return str(o)
    if isinstance(o, Path): 
        return str(o)
    if isinstance(o, set):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_builtin(v) for v in o]
    return o

def flatten_value_column(df: DataFrame):
    df_flat = df.select("*", "value.cpu", "value.heap", "value.state", "value.temp").drop("value")
    return df_flat



def create_spark_session(name):
    try:
        spark = SparkSession.builder \
            .appName(name) \
            .master("local[48]") \
            .config("spark.driver.memory", "100g") \
            .config("spark.driver.maxResultSize", "8g") \
            .config("spark.sql.shuffle.partitions", "192") \
            .config("spark.default.parallelism", "192") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.parquet.enableVectorizedReader", "true") \
            .config("spark.sql.parquet.columnarReaderBatchSize", "8192") \
            .config("spark.sql.files.maxPartitionBytes", str(256 * 1024 * 1024)) \
            .config("spark.ui.showConsoleProgress", "true") \
            .config("spark.sql.session.timeZone", "Asia/Ho_Chi_Minh") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:MaxGCPauseMillis=200") \
            .config("spark.memory.fraction", "0.7") \
            .config("spark.memory.storageFraction", "0.4") \
            .config("spark.rdd.compress", "true") \
            .config("spark.sql.autoBroadcastJoinThreshold", "50m") \
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
            .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
            .config("spark.sql.inMemoryColumnarStorage.batchSize", "20000") \
            .getOrCreate()
        spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoint")
        return spark
    except Exception as e:
        logger.error(f"Failed to create Spark session: {str(e)}")
        raise e

def time_logger(span="sec"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                if span == "sec":
                    print(f"Total processing time {elapsed:.0f} seconds")
                elif span == "min":
                    print(f"Total processing time {elapsed/60:,.2f} minutes")
                else:
                    print("Please define 'sec' or 'min' for logging")
        return wrapper
    return decorator

@time_logger(span="min")
def export_data(df, export_path, mode="overwrite"):
    try:
        df.write.mode(mode).parquet(export_path)
        logger.info(f"DATA EXPORTED SUCCESSFULLY TO {export_path}")
    except Exception as e:
        logging.error(f"FAILED TO EXPORT DATA: {str(e)}")
        raise e


#=================EVALUATION REPORT ARTIFACTS============================

def plot_roc_pr(y, proba, out):
    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, thr = precision_recall_curve(y, proba)
    #max f1 thr
    f1_scores = 2 * (prec * rec) / (prec + rec)
    max_f1_idx = f1_scores.argmax()
    max_f1_thr = thr[max_f1_idx]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    ax[0].plot([0,1],[0,1],'--r'); ax[0].legend(); ax[0].set_title("ROC")

    ax[1].plot(rec, prec, label=f"AUC={auc(rec,prec):.3f}\nMax F1={f1_scores[max_f1_idx]:.3f} at thr={max_f1_thr:.3f}")
    ax[1].legend(); ax[1].set_title("PR")
    fig.savefig(out)
    plt.close()

def plot_cm(y, pred, out):
    disp = ConfusionMatrixDisplay(confusion_matrix(y, pred))
    disp.plot(colorbar=False)
    plt.savefig(out)
    plt.close()

def plot_imp(model, names, out, top=30):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top]
    plt.figure(figsize=(8,6))
    plt.barh(range(top)[::-1], imp[idx])
    plt.yticks(range(top)[::-1], names[idx])
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def return_percentile_gain_chart(pred_df,true_col="y_true",y_pred="y_pred",y_proba="y_proba",number_of_thresholds=10,save_fig=False,output_dir=None,plot_name=None):
    df=pred_df.copy().sort_values(by=y_proba,ascending=False).reset_index(drop=True)

    df['decile']=pd.qcut(df.index,q=number_of_thresholds,labels=False,duplicates="drop")+1
    
    total_events=df[true_col].sum()
    gain_table=df.groupby('decile').agg(
        No_of_Observations=(true_col,'count'),
        Number_of_Events=(true_col,'sum'),
        Non_Events=(true_col,lambda x:(x==0).sum()),
        y_pred_proba_max =(y_proba, 'max'),
        y_pred_proba_min =(y_proba, 'min')
    ).reset_index()

    gain_table['Cumulative_Events']=gain_table.Number_of_Events.cumsum()
    gain_table['Cumulative_Gain (%)']=(gain_table['Cumulative_Events']/total_events*100).round(2)
    
    gain_table['Cumulative_Observations']=gain_table.No_of_Observations.cumsum()
    gain_table['Precision_Pct'] = (gain_table['Cumulative_Events']/gain_table['Cumulative_Observations']*100).round(2)
    gain_table['decile_low']=number_of_thresholds+1-gain_table['decile']

    bar_table=gain_table.sort_values(by='decile_low')

    fig,ax1= plt.subplots(figsize=(18,6))
    x=bar_table['decile_low']
    ax1.bar(x,bar_table['Number_of_Events'],label='Down State',color='skyblue',alpha=0.6)
    ax1.bar(x,bar_table['Non_Events'],bottom=bar_table['Number_of_Events'],label='Normal State',color='orange',alpha=0.6)
    ax1.set_xlabel('Decile (Low -> High Probability of Down)')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x.astype(str),fontsize=8)
    ax1.legend(bbox_to_anchor=(1.22,0.85),loc='upper right')

    ax2=ax1.twinx()
    x_line=number_of_thresholds+1-gain_table['decile']
    ax2.plot(x_line,gain_table['Cumulative_Gain (%)'],marker='o',color='red',label='Cumulative Coverage (%)')
    random_line=x_line*(gain_table['Cumulative_Gain (%)'].max()/x_line.max())
    ax2.plot(x_line,random_line,linestyle='--',color='grey',label='Random Model')
    ax2.plot(x,bar_table['Precision_Pct'],marker='s',linestyle='-',color='green',label='Cumulative Precision (%)')
    ax2.set_ylabel('Percentage')

    # Annotate
    for xi,yi in zip(x_line,gain_table['Cumulative_Gain (%)']):
        ax2.text(xi,yi+1.1,f"{yi:.2f}%",ha='center',va='bottom',fontsize=10,color='red')
    for xi,yi in zip(x,bar_table['Precision_Pct']):
        ax2.text(xi,yi+1.1,f"{yi:.2f}%",ha='center',va='bottom',fontsize=10,color='green')

    ax2.legend(bbox_to_anchor=(1.22,1),loc='upper right')
    if plot_name:
        plt.title(plot_name)
    plt.tight_layout()
    # plt.show()

    # Save fig
    if save_fig and output_dir:
        os.makedirs(output_dir,exist_ok=True)
        fig.savefig(os.path.join(output_dir,plot_name.replace(' ','_').lower()+'.png'),bbox_inches='tight',dpi=300)
        bar_table.to_csv(os.path.join(output_dir,plot_name.replace(' ','_').lower()+'_bartab.csv'),index=False)


def load_and_normalize_parquet(spark, file_path, schema, value_schema):
    try:
        df = spark.read.parquet(file_path)

        if "value" in df.columns:
            dtype = df.schema["value"].dataType
            
            # if string JSON -> convert struct
            if isinstance(dtype, StringType):
                print(f"Converting string 'value' → struct for file: {file_path}")
                df = df.withColumn(
                    "value",
                    F.from_json(F.col("value"), value_schema)
                )

            # If struct keep it and apply struct schema
            elif isinstance(dtype, StructType):
                # print(f"Filtering struct 'value' fields for file: {file_path}")
                existing_fields = [f.name for f in dtype.fields]
                target_fields = ["cpu", "heap", "state", "temp"]
                
                selected_fields = [
                    F.col(f"value.{f}").alias(f) if f in existing_fields else F.lit(None).alias(f)
                    for f in target_fields
                ]
                df = df.withColumn("value", F.struct(*selected_fields))

            else:
                print(f"Unexpected type for 'value' in {file_path}: {dtype}")
        else:
            print(f"Missing 'value' column in {file_path}")

        # Apply schema
        df = df.filter(F.col("service_name")=="check_juniper_component") 
        df = df.filter(F.col("component_name").rlike(r"(?i)fpc|routing engine"))
        df = df.select([F.col(c) if c in df.columns else F.lit(None).cast(t.dataType).alias(c)
                        for c, t in zip(schema.fieldNames(), schema.fields)])
        df = df.select(*[c for c in df.columns if c in schema.fieldNames()])
        return df

    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return None
    



def load_and_normalize_parquet_v2(spark, file_path, schema, value_schema):
    if len(file_path) == 0:
        return None
    try:
        df = spark.read.parquet(*file_path)
        if "value" in df.columns:
            dtype = df.schema["value"].dataType

            if isinstance(dtype, StringType):
                print(f"Converting string 'value' → struct for file: {file_path}")
                df = df.withColumn(
                    "value",
                    F.from_json(F.col("value"), value_schema)
                )

            elif isinstance(dtype, StructType):
                selected_fields = []
                existing_fields = [f.name for f in dtype.fields]

                for field in value_schema.fields:
                    if field.name in existing_fields:
                        selected_fields.append(
                            # F.col(f"value.{field.name}").cast(field.dataType).alias(field.name)
                            F.expr(f"try_cast(`value`.`{field.name}` AS {field.dataType.simpleString()})").alias(field.name)
                        )
                        
                    else:
                        selected_fields.append(
                            F.lit(None).cast(field.dataType).alias(field.name)
                        )
                df = df.withColumn("value", F.struct(*selected_fields))

            else:
                print(f"Unexpected type for 'value' in {file_path}: {dtype}")
        else:
            print(f"Missing 'value' column in {file_path}")

        # --- Ép toàn bộ schema về dạng chuẩn ---
        for field in schema:
            if field.name not in df.columns:
                df = df.withColumn(field.name, F.lit(None).cast(field.dataType))
            else:
                # df = df.withColumn(field.name, F.col(field.name).cast(field.dataType))
                df = df.withColumn(
                    field.name,
                    F.expr(f"try_cast(`{field.name}` AS {field.dataType.simpleString()})")
                )

        # --- Giữ đúng thứ tự các cột trong schema ---
        df = df.select([field.name for field in schema])
        return df

    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return None
    

def find_value_type_files(spark, files):
    """
    Tìm tất cả file parquet theo pattern, phân loại file có cột 'value'
    là StringType hoặc StructType.

    Args:
        path_pattern (str): ví dụ f"{SNMP_DIR}/month=10/day=*/*.parquet"

    Returns:
        tuple(list, list, list): (string_value_files, struct_value_files, error_files)
    """


    string_value_files = []
    struct_value_files = []
    error_files = []

    for f in files:
        try:
            df = spark.read.parquet(f)
            if "value" not in df.columns:
                continue

            dtype = df.schema["value"].dataType
            if isinstance(dtype, StringType):
                string_value_files.append(f)
            elif isinstance(dtype, StructType):
                struct_value_files.append(f)
            else:
                error_files.append((f, f"Unexpected type: {dtype.simpleString()}"))

        except Exception as e:
            error_files.append((f, str(e)))

    return string_value_files, struct_value_files, error_files
