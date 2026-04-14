import pandas as pd
import os
import sys
import re
import logging
from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pytz
import json
vntz=pytz.timezone('Asia/Ho_Chi_Minh')
import glob
import pickle
import time
from multiprocessing import Pool, cpu_count
import hashlib

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)
from model_architecture.helper_utils import time_logger, connect_to_mongo, read_records_to_dataframe
from model_architecture.vocab_builder import DeviceVocab, WordVocab
from model_architecture.preprocessing.sequence_mp import generate_sequences_mp
from dotenv import load_dotenv
load_dotenv(os.path.join(CUR_DIR, ".env"))

LOGGING_DIR = os.path.join(CUR_DIR, "logs")
os.makedirs(LOGGING_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S",
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout), 
                              logging.FileHandler(os.path.join(LOGGING_DIR, f"map_v4_label_clusterid_{datetime.now(vntz).strftime('%Y%m%d_%H%M%S')}.log"))]
                              )
logger = logging.getLogger(__name__)

#=============================RAW DATA PATHS=============================
FINAL_LAB_DIR = os.path.join(CUR_DIR, "data/sample_data/final/label")
# FINAL_INPUT_DIR = os.path.join(CUR_DIR, "data/sample_data/final/input")
FINAL_INPUT_DIR = r"/home/ubuntu/raw/noc-syslog-2"

noc_net_mapped = os.path.join(CUR_DIR, "data/noc_net_mapped/noc_net_data_v2.parquet")
ip_bras_file = os.path.join(CUR_DIR, "data/ip_bras_list/bras_ip.json")

OUTPUT_PKL_DIR = os.path.join(CUR_DIR, "output_minhnd74_v10.3/process_pkl_data_v4/")
VOCAB_PATH = os.path.join(CUR_DIR, "output_minhnd74_v10.3/vocab.pkl")
DEV_VOCAB_PATH = os.path.join(CUR_DIR, "output_minhnd74_v10.3/dev_vocab.pkl")

START_DATE = "2025-11-21"
END_DATE = "2026-01-01"

mongo_address = os.getenv("mongo_address", "")
mongo_user = os.getenv("mongo_user", "")
mongo_secret = os.getenv("mongo_secret", "")
mongo_db_name = os.getenv("mongo_db_name", "")
mongo_collection = os.getenv("mongo_collection", "label_v1")

if not os.path.exists(VOCAB_PATH) or not os.path.exists(DEV_VOCAB_PATH):
    logger.error(f"Vocab files not found! Check paths:\n{VOCAB_PATH}\n{DEV_VOCAB_PATH}")
    raise FileNotFoundError("Required vocab files are missing")

def read_parquet_with_schema(files: list):
    # pick one random file and check the dirname for schema
    sample_file = files[0]
    
    if os.path.dirname(sample_file).endswith("snmp_line_card_down_label"):
        df = pd.concat([pd.read_parquet(f, columns=["ip-address", "time"]) for f in files], ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
        df["time"] = df["time"].dt.tz_localize("Asia/Ho_Chi_Minh").dt.tz_convert("UTC")
        df = df.rename(columns={"ip-address": "ip_address"})
        return df
    
    elif os.path.dirname(sample_file).endswith("snmp_subscriber_label"):
        df = pd.concat([pd.read_parquet(f, columns=["ip-address", "time"]) for f in files], ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
        df["time"] = df["time"].dt.tz_localize("Asia/Ho_Chi_Minh").dt.tz_convert("UTC")
        df = df.rename(columns={"ip-address": "ip_address"})
        return df

    elif os.path.dirname(sample_file).endswith("snmp_traffic_label"):
        df = pd.concat([pd.read_parquet(f, columns=["bras_ip", "time"]) for f in files], ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
        df["time"] = df["time"].dt.tz_localize("Asia/Ho_Chi_Minh").dt.tz_convert("UTC")
        df = df.rename(columns={"bras_ip": "ip_address"})
        return df
    
    elif os.path.dirname(sample_file).endswith("snmp_trap_link_down_label"):
        df = pd.concat([pd.read_parquet(f, columns=["host", "@timestamp"]) for f in files], ignore_index=True)
        df["@timestamp"] = pd.to_datetime(df["@timestamp"])
        df = df.rename(columns={"host": "ip_address", "@timestamp": "time"})
        return df
    
    elif os.path.dirname(sample_file).endswith("syslog_error_label"):
        df = pd.concat([pd.read_parquet(f, columns=["host", "@timestamp"]) for f in files], ignore_index=True)
        df["@timestamp"] = pd.to_datetime(df["@timestamp"])
        df = df.rename(columns={"host": "ip_address", "@timestamp": "time"})
        return df
    else:
        logger.error("Unknown label file schema.")
        raise ValueError("Cannot determine schema for the provided label files.")

def load_label_local(shortlist_ip, start_date = START_DATE, end_date = END_DATE, FINAL_LAB_DIR = FINAL_LAB_DIR):
    start_time = pd.to_datetime(start_date, format="%Y-%m-%d")
    end_time = pd.to_datetime(end_date, format="%Y-%m-%d") + pd.Timedelta(hours=1)

    # convert to UTC for filtering
    start_time_utc = vntz.localize(start_time).astimezone(pytz.UTC)
    end_time_utc = vntz.localize(end_time).astimezone(pytz.UTC)

    # append labels
    label_type_dirs = os.listdir(FINAL_LAB_DIR)
    df_label_list = []
    
    for label_type in label_type_dirs:
        label_type_path = os.path.join(FINAL_LAB_DIR, label_type)
        if os.path.isdir(label_type_path):
            label_files = glob.glob(os.path.join(label_type_path, "**", "*.parquet"), recursive=True)
            if label_files:
                df_temp = read_parquet_with_schema(label_files)
                df_temp["label_type"] = label_type
                logger.info(f"Loading labels from {label_type}, total records: {len(df_temp)}")
                df_label_list.append(df_temp)
                print("\n")
    
    if not df_label_list:
        logger.error("No label files found!")
        raise ValueError("No label data available")
    
    df_label = pd.concat(df_label_list, ignore_index=True)
    
    # filter by shortlist_ip and time range
    df_label = df_label[df_label["ip_address"].isin(shortlist_ip)].drop_duplicates()
    df_label = df_label[
        (df_label["time"] >= start_time_utc) & 
        (df_label["time"] <= end_time_utc)
    ]
    
    logger.info(f"Total label records after filtering: {len(df_label)}")
    return df_label

def load_label_mongo(start_date = START_DATE, end_date = END_DATE, mongo_collection = mongo_collection):
    start_time = pd.to_datetime(start_date, format="%Y-%m-%d")
    end_time = (pd.to_datetime(end_date, format="%Y-%m-%d") + pd.Timedelta(hours=1))

    start_dt = vntz.localize(start_time).astimezone(pytz.UTC)
    end_dt = vntz.localize(end_time).astimezone(pytz.UTC)
    print(f"Start datetime (UTC): {start_dt}")
    print(f"End datetime (UTC): {end_dt}")

    filter_query = {
        "time": {
            "$gte": start_dt,
            "$lte": end_dt
        }
    }
    print(f"Filter query for MongoDB: {filter_query}")

    db = connect_to_mongo(mongo_address, mongo_user, mongo_secret, mongo_db_name)
    df_label = read_records_to_dataframe(
        database=db,
        collection_name=mongo_collection,
        filter_query=filter_query,
    )[["ip_address", "time"]].drop_duplicates()

    df_label = df_label[df_label["ip_address"].notnull()].copy()

    df_label["time"] = df_label["time"].dt.tz_localize("UTC")

    distinct_ips = df_label["ip_address"].unique().tolist()
    print("[LABEL] - Label data time range:")
    print(f"From {df_label['time'].min()} to {df_label['time'].max()}")
    print(f"Total label records from MongoDB: {len(df_label)}")

    return df_label, distinct_ips

ORIG_D3 = os.path.join(os.path.dirname(CUR_DIR), "DeviceIncidents", "drain3")
MERGED_OUTPUT = os.path.join(CUR_DIR, "data/sample_data/final/remapped")
os.makedirs(MERGED_OUTPUT, exist_ok=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3_REPO = os.path.join(ROOT, "AnomalyDetection", "drain3_training", "drain3_multiprocess")
sys.path.insert(0, str(D3_REPO))

from drain3 import LengthShardedTemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import LogMasker


def masking_data(df_local: pd.DataFrame) -> pd.DataFrame:
    config_temp = TemplateMinerConfig()
    config_temp.load(os.path.join(ORIG_D3, "drain3_v2.ini"))
    config_temp.profiling_enabled = False
    masker = LogMasker(config_temp.masking_instructions, "<", ">")
    extra_delimiters = config_temp.drain_extra_delimiters
    print(f"Extra delimiters: {extra_delimiters}")

    TRANSLATE_TABLE = str.maketrans(
        {ord(c): " " for c in extra_delimiters}
    )
    
    start_time = time.time()
    # Use multiprocessing to mask messages
    num_cores = max(1, cpu_count())
    print(f"Using {num_cores} cores for masking...")
    with Pool(num_cores) as pool:
        df_local["message"] = pool.map(masker.mask, df_local["message"], chunksize=300)

    end_time_1 = time.time()
    print(f"Masking completed in {(end_time_1 - start_time)/60:,.2f} minutes.")
    
    df_local["message"] = (
        df_local["message"]
        .astype(str)
        .str.strip()
        .str.translate(TRANSLATE_TABLE)
    )
    end_time_2 = time.time()
    print(f"Translation and stripping completed in {(end_time_2 - end_time_1)/60:,.2f} minutes.")
    
    unique_masked_msg = df_local["message"].unique().tolist()
    end_time_3 = time.time()
    print(f"Dropping duplicates completed in {(end_time_3 - end_time_2)/60:,.2f} minutes.")

    return df_local, unique_masked_msg

@time_logger(time_span="min")
def main():

    # =============================MAPPING_CLUSTER_ID=============================
    # Reuse the original DeviceIncidents/drain3 assets to avoid duplicating binaries

    vocab = WordVocab.load_vocab(VOCAB_PATH)
    device_vocab = DeviceVocab.load_vocab(DEV_VOCAB_PATH)
    logger.info(f"Loaded vocab with size: {len(vocab)} and device vocab size: {len(device_vocab)}")


    cfg = TemplateMinerConfig()
    cfg.load(os.path.join(ORIG_D3, "drain3_v2_1.ini"))
    # inference-only: no disk writes, no profiling
    cfg.snapshot_interval_minutes = 0
    cfg.profiling_enabled = False
    # enable max workers
    cfg.drain_shard_workers = os.cpu_count() * 3 // 2 if os.cpu_count() else 4
    logger.info(f"Using Drain3 with {cfg.drain_shard_workers} workers for sharded processing")

    miner = LengthShardedTemplateMiner(FilePersistence(os.path.join(ORIG_D3, "drain3_state_m_11_12_v2__12_batch_21_01.bin")), config=cfg)

    # =============================LOAD AND PREPROCESS DATA=============================
    START_DATE_STR = pd.to_datetime(START_DATE, format="%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S")
    END_DATE_STR = pd.to_datetime(END_DATE, format="%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"START DATE: {START_DATE_STR}, END DATE: {END_DATE_STR}")

    # shortlist IP
    shortlist_ip = pd.read_parquet(noc_net_mapped).ip_address.unique().tolist()
    ip_bras_list = json.load(open(ip_bras_file, "r"))
    shortlist_ip = list(set(shortlist_ip + ip_bras_list))

    # load label data
    # df_label = load_label_local(spark, shortlist_ip, start_date=START_DATE, end_date=END_DATE, FINAL_LAB_DIR=FINAL_LAB_DIR)
    df_label_org, _ = load_label_mongo(start_date=START_DATE, end_date=END_DATE, mongo_collection=mongo_collection)
    logger.info(f"Total distinct IPs in shortlist after merging with MongoDB labels: {len(shortlist_ip)}")
    df_label_org.drop(columns = "agg_status", errors='ignore', inplace=True)

    # main input data
    input_files = glob.glob(os.path.join(FINAL_INPUT_DIR,"**", "*.parquet"), recursive=True)
    input_files = [f for f in input_files if os.path.basename(os.path.dirname(f)) >= START_DATE and os.path.basename(os.path.dirname(f)) <= END_DATE]
    min_date = os.path.basename(os.path.dirname(min(input_files)))
    max_date = os.path.basename(os.path.dirname(max(input_files)))
    logger.info(f"[INPUT DATA] Full input data list: {min_date} to {max_date} - Total files: {len(input_files)}")

    hour_interval = 4 # hours
    time_stamp_skip = pd.to_datetime(START_DATE, format="%Y-%m-%d %H:%M:%S")
    for day_str in pd.date_range(start=START_DATE, end=END_DATE, freq="D").strftime("%Y-%m-%d"):
        short_listed_files = [f for f in input_files if os.path.basename(os.path.dirname(f)) == day_str]
        if not short_listed_files:
            logger.warning(f"[INPUT DATA] No input files found for date: {day_str}")
            continue
        for start_hour in range(0, 24, hour_interval):
            time_start = pd.to_datetime(f"{day_str} {start_hour:02d}:00:00", format="%Y-%m-%d %H:%M:%S")
            if time_start < time_stamp_skip:
                logger.info(f"Skipping {day_str} hour {start_hour:02d} as it's before start date-hour")
                continue

            start_time = time.time()
            end_hour = start_hour + hour_interval - 1
            pattern = re.compile(r"^data_(\d{2})-\d{2}-\d{2}\.parquet$")
            day_hour_files = [f for f in short_listed_files if int(pattern.search(os.path.basename(f)).group(1)) >= start_hour and int(pattern.search(os.path.basename(f)).group(1)) <= end_hour]
            if not day_hour_files:
                logger.warning(f"[INPUT DATA] No input files found for hours: {start_hour:02d} to {end_hour:02d} on {day_str}")
                continue
            logger.info(f"[INPUT DATA] Loading {len(day_hour_files)} files for hours: {start_hour:02d} to {end_hour:02d} on {day_str}")

            df_input = pd.concat([pd.read_parquet(f, columns=["host", "message", "@timestamp"]) for f in day_hour_files], ignore_index=True)
            
            if df_input["@timestamp"].dtype == "object":
                logger.info("[INPUT DATA] Converting @timestamp from object to datetime")
                df_input["@timestamp"] = pd.to_datetime(df_input["@timestamp"], errors="coerce", utc=True)
            df_input = df_input[df_input["host"].isin(shortlist_ip)].copy()
            df_input = df_input[df_input["@timestamp"].notnull()].copy()
            df_input.rename(columns={"@timestamp": "timestamp",
                                     "host": "ip_address"}, inplace=True)
            logger.info(f"[INPUT DATA] Time range after filtering: From {df_input['timestamp'].min()} to {df_input['timestamp'].max()}")
            df_input["ts5"] = df_input["timestamp"] + pd.Timedelta(minutes=5)

            # label merging
            df_label = df_label_org[df_label_org["ip_address"].isin(shortlist_ip)].copy()
            df_label["label"] = 1  # all labels are positive class
            start_time_utc = vntz.localize(pd.to_datetime(f"{day_str} {start_hour:02d}:00:00")).astimezone(pytz.UTC)
            end_datetime = pd.to_datetime(f"{day_str} {end_hour:02d}:59:59") + pd.Timedelta(hours=2)
            end_time_utc = vntz.localize(end_datetime).astimezone(pytz.UTC)
            logger.info(f"[FILTER] SET condition for label from {start_time_utc} to {end_time_utc}")
            
            df_label = df_label[(df_label["time"] >= start_time_utc) & (df_label["time"] <= end_time_utc)].copy()
            logger.info(f"[LABEL DATA] ACTUAL Time range for labels after filtering: From {df_label['time'].min()} to {df_label['time'].max()} - Total records: {len(df_label)}")
            df_input = df_input.sort_values(by=["ts5", "ip_address"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
            df_label = df_label.sort_values(by=["time", "ip_address"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

            df_merged = pd.merge_asof(
                df_input,
                df_label,
                left_on="ts5",
                right_on="time",
                by="ip_address",
                direction="forward",
                tolerance=pd.Timedelta(minutes=15)
            )
            
            df_input["label"] = df_merged["label"].fillna(0).astype(int)
            del df_merged

            logger.info(f"[MERGE] - After dropping close events: Total records: {len(df_input)}")
            df_input = df_input[["ip_address", "timestamp", "label", "message"]].drop_duplicates().reset_index(drop=True)
            df_input.rename(columns={"ip_address": "ip"}, inplace=True)
            logger.info(f"[MERGE] - Final label distribution:\n{df_input['label'].value_counts(normalize=True)}")
            logger.info(f"[MERGE] - Date time range after merging: From {df_input['timestamp'].min()} to {df_input['timestamp'].max()}")

            df_input.dropna(subset=["message"], inplace=True)

            # =============================DRAIN3 BATCH MATCHING=============================

            val_check1 = df_input.shape[0]
            df_input, unique_masked_msg = masking_data(df_input)
            len_uni_mask = len(unique_masked_msg)
            logger.info(f"[MASK] Total unique masked messages for batch matching: {len_uni_mask} vs total rows: {val_check1}")
            val_check2 = df_input.shape[0]
            if val_check1 != val_check2:
                logger.error(f"Row count changed after masking! Before: {val_check1}, After: {val_check2}")
                raise ValueError("Row count mismatch after masking")

            res_matches = miner.batch_match(unique_masked_msg)
            num_matched = sum(1 for m in res_matches if m is not None)
            num_unmatched = len(res_matches) - num_matched
            logger.info(f"[DRAIN3] Batch matching completed. Total: {len(res_matches)}, Matched: {num_matched}, Unmatched: {num_unmatched}")

            # create a dict for faster lookup - handle None matches
            msg_to_match = {
                msg: (match.get_template() if match is not None else None)
                for msg, match in zip(unique_masked_msg, res_matches)
            }
            logger.info(f"[DRAIN3] Sample of message to template mapping:")
            logger.info(f"{list(msg_to_match.items())[:5]}")

            df_input["log_key"] = df_input["message"].map(msg_to_match)
            df_input.drop(columns=["message"], inplace=True)
            df_input["log_key"] = df_input["log_key"].apply(lambda x: int(vocab.stoi.get(x, vocab.unk_index)))

            # =============================FINAL OUTPUT=============================
            logger.info(f"[FINAL] Final Schema after mapping cluster IDs:")
            logger.info(f"{df_input.info()}")
            logger.info(f"Sample data:\n{df_input.head(5)}")
            FINAL_OUT_DIR = os.path.join(MERGED_OUTPUT, "seqs_latest_v4.parquet", f"date={day_str}")
            os.makedirs(FINAL_OUT_DIR, exist_ok=True)
            logger.info(f"WRITING OUTPUT TO {FINAL_OUT_DIR}")
            
            file_name = f"data_{start_hour:02d}-{end_hour:02d}.parquet"
            df_input.to_parquet(os.path.join(FINAL_OUT_DIR, file_name), index=False)
            logger.info(f"Saved mapped data to {os.path.join(FINAL_OUT_DIR, file_name)}")
            del df_input
            end_time = time.time()
            total_time = (end_time - start_time)/60
            logger.info(f"Completed processing for {day_str} hours {start_hour:02d}-{end_hour:02d} in {total_time:,.2f} minutes.")

    logger.info(f"START CONVERTING TO SEQUENCES PICKLE FORMAT")
    TEST_DIR = os.path.join(MERGED_OUTPUT, "seqs_latest_v4.parquet")
    part_folders = [d for d in os.listdir(TEST_DIR) if d.startswith("date=")]
    part_folders = sorted(part_folders, key=lambda x: x.split("=")[1])  # keep partition order deterministic
    dates = [d.split("=")[1] for d in part_folders]
    logger.info(f"Total unique dates: {len(dates)}")

    for i in range(0, len(dates), 1):
        date_chunk = dates[i:i+1]
        logger.info(f"PROCESSING FOR {' and '.join(date_chunk)}")

        files = []
        for d in date_chunk:
            files.extend(glob.glob(os.path.join(TEST_DIR, f"date={d}", "*.parquet")))

        if not files:
            continue
        
        pattern = re.compile(r"^data_(\d+)-\d+\.parquet$")
        files_sorted = sorted(files, key = lambda x: int(pattern.match(os.path.basename(x)).group(1)))

        for file in files_sorted:
            
            out_date = date_chunk[-1]
            out_time = os.path.basename(file).split(".")[0]
            logger.info(f"READING FILE FROM {out_date}/{out_time}")
            df = pd.read_parquet(file, columns = ["ip", "timestamp", "log_key", "label"])
            logger.info(df["label"].value_counts(normalize=True))

            seqs, _ = generate_sequences_mp(
                df,
                window_size=15,
                mode="time_window",
                device_vocab=device_vocab,
                vocab=vocab,
                max_workers=os.cpu_count() * 2 // 3
            )
            del df

            if not seqs:
                continue 

            logger.info(f"Sample seq for batch: {seqs[1000]}")

            FINAL_OUT = os.path.join(OUTPUT_PKL_DIR, f"{out_date}", f"{out_time}.pkl")
            os.makedirs(os.path.dirname(FINAL_OUT), exist_ok=True)
            with open(FINAL_OUT, "wb") as f:
                pickle.dump(seqs, f)
            logger.info(f"Saved sequences pickle to {FINAL_OUT}")



if __name__ == "__main__":
    main()
