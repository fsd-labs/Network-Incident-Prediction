import logging
import os, sys
from pathlib import Path
import argparse
import glob
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NOC_NET_MAPPED = str(PROJECT_ROOT / "data" / "noc_net_mapped" / "noc_net_data_v2.parquet")
IP_BRAS_FILE = str(PROJECT_ROOT / "data" / "ip_bras_list" / "bras_ip.json")
OUTPUT_DEV_VOCAB = str(PROJECT_ROOT / "output_minhnd74_v10.3" / "dev_vocab.pkl")

from model_architecture.vocab_builder import DeviceVocab
from model_architecture.helper_utils import create_spark_session
from map_label_clusterid import load_label_mongo, START_DATE, END_DATE, mongo_collection
import json
from pyspark.sql import functions as F
import pandas as pd

def create_dev_vocab():
    spark = create_spark_session("DevVocabExecutor", show_progress=True, enable_ui=True)

    # shortlist IP
    shortlist_ip = spark.read.parquet(NOC_NET_MAPPED).select(F.collect_set("ip_address")).collect()[0][0]
    ip_bras_list = json.load(open(IP_BRAS_FILE, "r"))
    shortlist_ip = list(set(shortlist_ip + ip_bras_list))
    logger.info(f"Total shortlist IP (with BRAS) addresses: {len(shortlist_ip)}")

    device_vocab = DeviceVocab(shortlist_ip)
    logger.info(f"Total device vocab size: {len(device_vocab.itos)}")

    device_vocab.save_vocab(OUTPUT_DEV_VOCAB)
    logger.info(f"Saved device vocab to {OUTPUT_DEV_VOCAB}")



