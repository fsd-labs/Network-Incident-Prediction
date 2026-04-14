import math
import re
from venv import logger
import pandas as pd
import os
import numpy as np
import jinja2
import numpy as np
import time
import pickle
from typing import Optional, List
from datetime import datetime, timedelta, timezone
from sklearn.metrics import (
    classification_report,
)
from urllib.parse import quote_plus
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient, DESCENDING
import dotenv

dotenv.load_dotenv('')


MONGO_URI = os.environ.get('mongo_address')
MONGO_PASS = os.environ.get('mongo_secret')
MONGO_USER = os.environ.get('mongo_user')
MONGO_DB = os.environ.get('mongo_db_name')
# MONGO_DB_PREDICTION_COLLECTION_NAME = os.environ.get('MONGO_DB_PREDICTION_COLLECTION_NAME')
# MONGO_COLLECTION_RESULTS = os.environ.get('MONGO_COLLECTION_RESULTS')

encoded_pass = quote_plus(MONGO_PASS)
mongo_uri = f"mongodb://{MONGO_USER}:{encoded_pass}@{MONGO_URI}"


def create_mongo_client():
    client = MongoClient(
        mongo_uri,
    )
    return client


def read_records_to_dataframe(collection_name, filter_query={}):
    """
    Read records and convert to pandas DataFrame.
    
    Args:
        collection_name: Name of the collection
        filter_query: Dictionary containing filter conditions (default: {})
    
    Returns:
        pd.DataFrame: DataFrame containing the records
    """
    try:
        client = create_mongo_client()
        db = client[MONGO_DB]
        collection = db[collection_name]
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

def get_label_old(start, end):
    logger.info(f"Using historical labels from the database.")
    start = start - timedelta(minutes=20)
    end = end + timedelta(minutes=20)
    
    label_filter_query = {
        "time": {
            "$gte": start,
            "$lte": end
        }
    }

    df_labels = read_records_to_dataframe(collection_name="label_v1", filter_query=label_filter_query)
    df_labels = df_labels[["ip_address", "time", "agg_status"]].drop_duplicates()
    df_labels["time"] = df_labels["time"].dt.tz_localize("UTC")
    min_lab_time = df_labels["time"].min().tz_convert('Asia/Ho_Chi_Minh')
    max_lab_time = df_labels["time"].max().tz_convert('Asia/Ho_Chi_Minh')
    logger.info(f"Label time range from {min_lab_time} to {max_lab_time}, total labels: {len(df_labels)}")
    return df_labels


if __name__ == "__main__":
    start_time = datetime(2026, 3, 16,1,0,0,tzinfo=timezone.utc)
    end_time = datetime(2026, 4, 1, 0,0,0,tzinfo=timezone.utc)
    df_labels = get_label_old(start_time, end_time)
    print(df_labels.head())

    df_nocmap = pd.read_parquet(r"/home/ubuntu/AnomalyDetection/dev_refactor/data/raw/noc_net_mapped/noc_net_data_v2.parquet")["ip_address"].unique().tolist()
    print(f"Total unique IPs in NOC map: {len(df_nocmap)}")
    df_labels = df_labels[df_labels["ip_address"].isin(df_nocmap)]

    OUT_DIR = "data/label"

    df_labels.to_parquet(f"{OUT_DIR}/labels_March_2026.parquet", engine="pyarrow", index=False)
    
    # #set tz to HCM
    # from zoneinfo import ZoneInfo
    # start_test = datetime(2026, 3, 15,0,0,0,tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    # end_test = datetime(2026, 4, 1,0,0,0,tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    # date_lst = pd.date_range(start=start_test, end=end_test, freq='D')
    # print(date_lst)

    # agg_ips_daily = {}
    # for date in date_lst:
    #     # for each date, get all ips within the latest 7 days (current date included)
    #     start_date = date - timedelta(days=7)
    #     end_date = date
    #     date = date - timedelta(days=1)
    #     print(f"Getting labels for date: {date.date()}, from {start_date.date()} to {end_date.date()}")
    #     df_labels_date = df_labels[(df_labels["time"] >= start_date) & (df_labels["time"] <= end_date)]
    #     ips_uni = df_labels_date["ip_address"].unique().tolist()
    #     agg_ips_daily[date.date()] = ips_uni
    #     print(f"From {df_labels_date['time'].min().tz_convert('Asia/Ho_Chi_Minh')} to {df_labels_date['time'].max().tz_convert('Asia/Ho_Chi_Minh')}, total unique IPs: {len(ips_uni)}")
    
    # OUT_DIR = "data/label"
    # os.makedirs(OUT_DIR, exist_ok=True)
    # import json
    # with open(os.path.join(OUT_DIR, "agg_ips_daily.json"), "w") as f:
    #     json.dump({str(k): v for k, v in agg_ips_daily.items()}, f, indent=2)
    #     print(f"Saved aggregated IPs daily to {os.path.join(OUT_DIR, 'agg_ips_daily.json')}")

        



