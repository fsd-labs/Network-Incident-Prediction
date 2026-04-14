import json
import re
from pyspark.sql import SparkSession
from pyspark import SparkConf
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import pandas as pd
import time
import json
import os
from dotenv import load_dotenv
import pyspark.sql.functions as F
from multiprocessing import Pool, cpu_count


load_dotenv()

PROTOCOL_ELK = os.environ.get('PROTOCOL_ELK')
HOST_ELK = os.environ.get('HOST_ELK')
USER_ELK = os.environ.get('USER_ELK')
PORT_ELK = os.environ.get('PORT_ELK')
SCRET_ELK = os.environ.get('SCRET_ELK')

es = Elasticsearch(
    f"{PROTOCOL_ELK}://{HOST_ELK}:{PORT_ELK}",
    http_auth=(USER_ELK, SCRET_ELK,),
    # timeout=60
)

index_name = "noc-physical-topology"
folder_path ='/home/ubuntu/raw/bras/'


def generate_script_queries(start, end):
    time_filter = {
        "range": {
            "@timestamp": {
                "gte": start,
                "lt": end
            }
        }
    }
    return {
            "_source": [
                "@timestamp",
                "local_device_name","local_device_ip","local_device_function",
                "remote_device_name","remote_device_ip","remote_device_function",
                "local_name","remote_name",
                "local_data.aggregate_name","remote_data.aggregate_name"
            ],
            "query": {
                "bool": {
                    "must": [
                        { "term": { "local_device_function.keyword": "Metro BroadBand BRAS" } },
                        { "term": { "remote_device_function.keyword": "Metro BroadBand Router Edge" } },
                        { "exists": { "field": "remote_device_name" } },
                        ],
                    "filter": [time_filter],
                }
            }
    }


def write_to_file(docs, time_value):
    if(len(docs) ==0):
        return

    print("Convert docs to dataframe!")
    df = pd.DataFrame(docs)

    file_name = f"{time_value.strftime('%Y-%m-%d')}.parquet"
    file_path = f"{folder_path}/{file_name}"

    df.to_parquet(file_path, engine="pyarrow", index=False)
    print(f"Done! Written {len(df)} rows to {file_path}")


def collect_daily(time_value):
    start = time_value.strftime("%Y-%m-%dT%H:%M:00.000+07:00")
    end = (time_value + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:00.000+07:00")
    query = generate_script_queries(start, end)
    
    # Scroll through all matching documents
    scroll_time = "2m"  # Adjust scroll time as needed
    docs = []
    response = es.search(index=index_name, body=query,
                             scroll=scroll_time, size=10000, request_timeout=30)

    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    # Collect all documents
    doc_count = 0 
    while hits:
        docs.extend(hits)
        doc_count += len(hits)
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        print(f"Collect {len(docs)}")

    docs =  [obj['_source'] for obj in docs]
    print("Lengt docs", len(docs))
    es.clear_scroll(scroll_id=scroll_id)
    write_to_file(docs=docs, time_value=time_value)


def worker(days):
    print(f"Processing {days.strftime('%Y-%m-%d %H:%M:%S')}")
    collect_daily(days)

def collect_oneoff(start_date, end_date):
    days = []
    current_time = start_date

    while current_time <= end_date:
        days.append(current_time)
        current_time += timedelta(days=1)

    pool_size = 4
    with Pool(processes=pool_size) as pool:
        pool.map(worker, days)



def main(start_date=None, end_date=None):
    start_date = datetime(2025, 11, 21).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime(2025, 12, 31).replace(hour=0, minute=0, second=0, microsecond=0)
    print("Collect oneoff!")
    collect_oneoff(start_date, end_date)
    print("Done")


if __name__ == "__main__":
    main()
