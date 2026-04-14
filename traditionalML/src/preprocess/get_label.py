# from pyspark.sql import SparkSession

# # Tạo SparkSession
# spark = SparkSession.builder.appName("CountParquetRecords").getOrCreate()

# # Đường dẫn thư mục chứa các file parquet
# parquet_dir = "/home/ast/AnomalyDetection/minhnd74/nlp_approach/test/data/input_data/full_test_2708/final_merged_data_nolvt"

# # Đọc toàn bộ parquet trong thư mục
# df = spark.read.parquet(parquet_dir)

# # Đếm số lượng bản ghi
# count = df.count()
# print(f"Số lượng bản ghi: {count}")


from elasticsearch import Elasticsearch
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from datetime import datetime, timedelta
import pandas as pd
import time
import json
import sys
import os
import pytz
from dotenv import load_dotenv
import logging
os.environ['TZ'] = 'Asia/Ho_Chi_Minh'
import time
time.tzset()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

PROTOCOL_ELK = os.environ.get('PROTOCOL_ELK')
HOST_ELK = os.environ.get('HOST_ELK')
USER_ELK = os.environ.get('USER_ELK')
PORT_ELK = os.environ.get('PORT_ELK')
SCRET_ELK = os.environ.get('SCRET_ELK')

tz_VN = pytz.timezone('Asia/Ho_Chi_Minh')

es = Elasticsearch(
    f"{PROTOCOL_ELK}://{HOST_ELK}:{PORT_ELK}",
    http_auth=(USER_ELK, SCRET_ELK,),
    timeout=60
)

def get_data():
    index = "scc_collector_noc_net-*"
    end = datetime(2026, 3, 30, 0, 0, 0).replace(hour=23, minute=59, second=0, microsecond=0)
    # start = end - timedelta(days=2)
    start = datetime(2025, 3, 15, 0, 0, 0).replace(hour=0, minute=0, second=0, microsecond=0)
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start.strftime("%Y-%m-%dT%H:%M:%S.000+07:00"),
                                "lt": end.strftime("%Y-%m-%dT%H:%M:%S.000+07:00"),
                            }
                        }
                    },
                    {
                        "term": {
                            "service_name.keyword" : "check_juniper_component"
                        }

                    }
                ],
                "must_not": [
                    {
                        "terms": {
                            "value.state.keyword": ["2", "7"]
                        }
                    }
                ]
            }
        },
        "sort": [
            {"@timestamp": {"order": "asc"}}
        ]
    }

    scroll_time = "2m"  # Adjust scroll time as needed
    docs = []
    response = es.search(index=index, body=query, scroll=scroll_time, size=10000, request_timeout=30)

    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    while hits:
        docs.extend(hits)
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']

    docs = [obj['_source'] for obj in docs]
    es.clear_scroll(scroll_id=scroll_id)


    docs = [obj for obj in  docs if "FPC" in obj['component_name'] or "Routing Engine" in obj["component_name"]]
    print(f"Total documents after filtering: {len(docs)}")

    docs = [{"ip_address": d.get("ip-address"), "time": d.get("@timestamp"), "agg_status": 'DOWN'} for d in docs]


    df = pd.DataFrame(docs)
    OUTPUT_DIR = ""
    df.to_parquet(f"{OUTPUT_DIR}/down_ip_March_2026.parquet", engine="pyarrow", index=False)

    logger.info(f"Collected {len(docs)} documents from index {index}")
    return docs

if __name__ == "__main__":
    get_data()