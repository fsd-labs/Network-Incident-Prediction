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

index_name = ""
folder_path =''
list_ip = [] 

def generate_script_queries(list_ip, start, end):
    # Giới hạn truy vấn theo start và end time
    time_filter = {
        "range": {
            "@timestamp": {
                "gte": start,
                "lt": end
            }
        }
    }
#    ip_queries = [{"match": {"host.keyword": ip}} for ip in list_ip]
    return {
        "query": {
            "bool": {
                "should": [], #ip_queries, #queries,
                "filter": [time_filter],
                #"minimum_should_match": 1
            }
        }
    }


def is_file_exists(time_value):
    folder_name = time_value.strftime("%Y-%m-%d")
    hour_str = time_value.strftime("%H-%M-%S")
    sub_folder = os.path.join(folder_path, folder_name)
    file_name = f"data_{hour_str}.parquet"
    file_path = os.path.join(sub_folder, file_name)

    if os.path.exists(file_path):
        print(f"File {file_path} exists!")
        return True

def write_to_file(docs, time_value):
    if(len(docs) ==0):
        # print(f"Not found any in {time_value}")
        return

    print("Convert docs to dataframe!")
    df = pd.DataFrame(docs)

    folder_name = time_value.strftime("%Y-%m-%d")
    hour_str = time_value.strftime("%H-%M-%S")
    sub_folder = os.path.join(folder_path, folder_name)
    os.makedirs(sub_folder, exist_ok=True)

    file_name = f"data_{hour_str}.parquet"
    file_path = os.path.join(sub_folder, file_name)

    # Tránh ghi đè: Nếu file đã tồn tại, tạo file mới có số đếm
    count = 1
    while os.path.exists(file_path):
        file_name = f"data_{hour_str}_{count}.parquet"
        file_path = os.path.join(sub_folder, file_name)
        count += 1

    df.to_parquet(file_path, engine="pyarrow", index=False)
    print(f"Done! Written {len(df)} rows to {file_path}")


def collect_hourly(time_value):
    if is_file_exists(time_value):
        return
    start = time_value.strftime("%Y-%m-%dT%H:%M:00.000+07:00")
    end = (time_value + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:00.000+07:00")
    # print("==>", start, end)

    #global patterns
    query = generate_script_queries(list_ip, start, end)
    #print(query)
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
    es.clear_scroll(scroll_id=scroll_id)
    write_to_file(docs=docs, time_value=time_value)
    # Close the scroll to free resources
    # print("Done", time_value)


def worker(hour):
    print(f"Processing {hour.strftime('%Y-%m-%d %H:%M:%S')}")
    collect_hourly(hour)

def collect_oneoff(start_date, end_date):
    hours = []
    current_time = start_date

    while current_time < end_date:
        hours.append(current_time)
        current_time += timedelta(hours=1)

    #for hour in hours:
    #    worker(hour=hour)
    #    time.sleep(1)
    pool_size = 3
    with Pool(processes=pool_size) as pool:
        pool.map(worker, hours)



def get_index_mapping(index_name):
    """
    Lấy mapping của một index trong Elasticsearch.

    Args:
        es_client: client kết nối Elasticsearch (ví dụ: Elasticsearch(...))
        index_name (str): Tên index muốn lấy mapping

    Returns:
        dict: Mapping chi tiết của index
    """
    try:
        mapping = es.indices.get_mapping(index=index_name)
        print(f"Mapping for index '{index_name}':")
        return mapping[index_name]['mappings']
    except Exception as e:
        print(f"Error getting mapping for index '{index_name}': {e}")
        return None


def main(start_date=None, end_date=None):
    # start_date = datetime(2026, 1, 15).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = datetime(2025,11, 21).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime(2025, 12, 31).replace(hour=0, minute=0, second=0, microsecond=0)
    # get_index_mapping(index_name = "nocnet-*")
    print("Collect oneoff!")
    collect_oneoff(start_date, end_date)


if __name__ == "__main__":
    main()
