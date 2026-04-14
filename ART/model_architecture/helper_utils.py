from functools import wraps
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
import pytz
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import pickle
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

vntz = pytz.timezone("Asia/Ho_Chi_Minh")

# read multiple pkl files
def multi_load_pkl(folder_path):
    files = glob.glob(os.path.join(folder_path, "**", "*.pkl"), recursive=True)
    seqs_all = []
    for file in tqdm(files, desc=f"Loading pkl from {folder_path}", total=len(files)):
        seqs = pickle.load(open(file, "rb"))
        seqs_all.extend(seqs)
    return seqs_all

# define time logger decorator
def time_logger(time_span="sec"):
    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print(e)
                result = None
            finally:
                end = time.time()
                total_time = end-start
                if time_span == "sec":
                    print(f"Total processing time {total_time:,.0f} seconds")
                elif time_span =="min":
                    print(f"Total processing time {total_time/60:,.2f} minutes")
                else:
                    print("Please define sec or min for logging")
            return result
        return inner
    return wrapper

def visualize_loss(loss_list, save_fig=True, output_dir="./"):
    loss_arr = np.asarray(loss_list).ravel()
    loss_df = pd.DataFrame({
        "epoch": np.arange(1, len(loss_arr) + 1),
        "train_loss": loss_arr
    })

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    fig.suptitle(f"Loss viz for {len(loss_arr)} epochs")
    sns.lineplot(data=loss_df, x="epoch", y="train_loss", ax=ax)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.tight_layout()

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        save_time = datetime.now(vntz).strftime("%d-%m-%y_%H-%M")
        out_path = os.path.join(output_dir, f"loss_viz_{save_time}.png")
        fig.savefig(out_path, dpi=150)
        print(f"Loss viz saved at {out_path}")

    plt.close(fig)

def write_log_txt(log_str, log_file, mode = "a"):
    with open(log_file, mode) as f:
        f.write(log_str + "\n")

def visualize_roc_auc(y_true, y_scores, save_fig, output_dir="./"):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc}")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    # plt.show()

    # save_fig
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        save_time = datetime.now(vntz).strftime("%d-%m-%y_%H-%M")
        fig.savefig(os.path.join(output_dir, f"roc_auc_viz_{save_time}"))
        print(f"ROC AUC viz saved at {os.path.join(output_dir, f'roc_auc_viz_{save_time}')}")
        plt.close()

def plot_cm(y, pred, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix(y, pred))
    disp.plot(colorbar=False)
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
    ).reset_index()

    gain_table['Cumulative_Events']=gain_table.Number_of_Events.cumsum()
    gain_table['Cumulative_Gain (%)']=(gain_table['Cumulative_Events']/total_events*100).round(2)
    
    gain_table['Cumulative_Observations']=gain_table.No_of_Observations.cumsum()
    gain_table['Precision_Pct'] = (gain_table['Cumulative_Events']/gain_table['Cumulative_Observations']*100).round(2)
    gain_table['decile_low']=number_of_thresholds+1-gain_table['decile']

    bar_table=gain_table.sort_values(by='decile_low')

    fig,ax1=plt.subplots(figsize=(18,6))
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
        fig.savefig(os.path.join(output_dir,plot_name.replace(' ','_').lower()+f'{datetime.now(vntz).strftime("%d-%m-%y_%H-%M")}.png'),bbox_inches='tight',dpi=300)

class EarlyStopper:
    def __init__(self, patience = 20, min_delta = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = None

    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False
    
def create_spark_session(appname, show_progress=True, enable_ui=True):
    builder = (
        SparkSession.builder
        .appName(appname)
        .master("local[36]")  # 12 threads, leave resources for Python multiprocessing
        .config("spark.driver.memory", "96g")  # 150GB for Spark, leave 150GB for Python/OS
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.5") #.config("spark.local.dir", "/mnt/spark1,/mnt/spark2")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.autoBroadcastJoinThreshold", "50MB")
        .config("spark.sql.shuffle.partitions", "24")  # 2x threads
        .config("spark.default.parallelism", "24")
        .config("spark.sql.files.maxPartitionBytes", str(256 * 1024 * 1024))
        .config("spark.sql.files.openCostInBytes", str(32 * 1024 * 1024))
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.parquet.columnarReaderBatchSize", "4096")
        .config("spark.sql.parquet.mergeSchema", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.inMemoryColumnarStorage.compressed", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=45 -XX:+ParallelRefProcEnabled")
        .config("spark.locality.wait", "1s")
        .config("spark.shuffle.file.buffer", "512k")
        .config("spark.reducer.maxSizeInFlight", "48m")
        .config("spark.shuffle.sort.bypassMergeThreshold", "200")
    )
    if show_progress:
        builder = builder.config("spark.ui.showConsoleProgress", "true")
    if enable_ui:
        builder = builder.config("spark.ui.enabled", "true")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ================================= MongoDB Helper Functions ==============================

from pymongo import MongoClient
from pymongo import UpdateOne
from pymongo.errors import ConnectionFailure

# Function to connect to MongoDB
def connect_to_mongo(mongo_address, mongo_user, mongo_secret, mongo_db_name):
    try:
        # Create the connection string
        mongo_uri = f"mongodb://{mongo_user}:{mongo_secret}@{mongo_address}"

        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        print("Connected to MongoDB successfully!")
        print("List databases: ", client.list_database_names())

        # Return the database object
        return client[mongo_db_name]
    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None


# Create a new collection
def create_new_collection(database, collection_name):
    if collection_name not in database.list_collection_names():
        try:
            database.create_collection(collection_name)
            print("Collection created successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
    return None


# Function to insert one record into a collection
def insert_one_record(database, collection_name, data):
    try:
        # Get the collection
        collection = database[collection_name]

        # Insert data
        result = collection.insert_one(data)
        print(f"Data inserted with ID: {result.inserted_id}")
    except Exception as e:
        print(f"Failed to insert data: {e}")
    return None


# Function to insert many records into a collection
def insert_many_records(database, collection_name, data):
    try:
        # Get the collection
        collection = database[collection_name]

        # Insert data
        result = collection.insert_many(data)
        print(f"Data inserted with length: {len(result.inserted_ids)}")
    except Exception as e:
        print(f"Failed to insert data: {e}")
    return None


def delete_list_collections(database, list_collection_names):
    for collection_name in list_collection_names:
        if collection_name in database.list_collection_names():
            database[collection_name].drop()
            print(f"Collection '{collection_name}' has been deleted.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    return None


def print_collection_data(database, collection_name):
    collection = database[collection_name]
    document_count = collection.count_documents({})
    print(f"Total documents in collection: {document_count}")
    documents = collection.find()
    print("All documents in the collection:")
    for doc in documents:
        print(doc)
    return None

def upsert_many_records(database, collection_name, data):
    """
    Args:
        database: MongoDB database object
        collection_name: Name of the collection
        data: List of dictionaries with 'ip' and 'timestamp' fields
    
    Returns:
        tuple: (upserted_count, modified_count)
    """
    try:        
        collection = database[collection_name]
        operations = []
        
        for record in data:
            operations.append(
                UpdateOne(
                    {"ip": record["ip"], "timestamp": record["timestamp"]},
                    {"$set": record},
                    upsert=True
                )
            )
        
        if operations:
            result = collection.bulk_write(operations)
            print(f"Upserted: {result.upserted_count} records, Modified: {result.modified_count} records")
            return result.upserted_count, result.modified_count
        else:
            print("No operations to perform")
            return 0, 0
            
    except Exception as e:
        print(f"Failed to upsert data: {e}")
        return 0, 0

def read_records_to_dataframe(database, collection_name, filter_query={}):
    """
    Read records and convert to pandas DataFrame.
    
    Args:
        database: MongoDB database object
        collection_name: Name of the collection
        filter_query: Dictionary containing filter conditions (default: {})
    
    Returns:
        pd.DataFrame: DataFrame containing the records
    """
    try:
        collection = database[collection_name]
        documents = list(collection.find(filter_query))
        df = pd.DataFrame(documents)
        
        # Remove MongoDB _id field if exists
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        print(f"Retrieved {len(df)} records as DataFrame from '{collection_name}'")
        return df
    except Exception as e:
        print(f"Failed to read data: {e}")
        return pd.DataFrame()