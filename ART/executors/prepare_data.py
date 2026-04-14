import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import pytz
import pandas as pd
import json

from model_architecture.helper_utils import create_spark_session
from model_architecture.preprocessing.sequence_mp import generate_sequences_mp
from model_architecture.vocab_builder import DeviceVocab, WordVocab
import warnings
from pyspark.sql import functions as F
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")


class SequencePreper:
    def __init__(self, args):
        self.args = args
        self.vocab = WordVocab.load_vocab(args.vocab_path)
        self.device_vocab: Optional[DeviceVocab] = None

    def _read_parquet(self, parquet_path: str, spark):
        df = spark.read.parquet(parquet_path)
        return df

    def _iter_pandas_partitions(self, sdf):
        partitions = sdf.rdd.glom()
        for partition_rows in partitions.toLocalIterator():
            if partition_rows:
                yield pd.DataFrame([row.asDict() for row in partition_rows])

    def _get_unique_device_vocab(self, sdf):
        device_set = []
        for row in sdf.select("ip").distinct().toLocalIterator():
            device_set.append(row.ip)
        device_series = pd.Series(device_set)
        device_vocab = DeviceVocab(device_series)
        return device_vocab

    def _process_data(self, data_path):
        args = self.args
        data_type = os.path.basename(data_path).split(".")[0]
        temp_dir = os.path.join(args.output_dir, f"temp_{data_type}")
        spark = create_spark_session("SequencePrep")
        if not os.path.exists(temp_dir):
            sdf = self._read_parquet(data_path, spark)
            sdf = sdf.withColumn("date", F.col("timestamp").cast("date"))

        try:
            if data_type == "train":
                if os.path.exists(args.device_vocab_path):
                    device_vocab = DeviceVocab.load_vocab(args.device_vocab_path)
                    self.device_vocab = device_vocab
                    logger.info(f"Loaded device vocab from {args.device_vocab_path}")
                else:
                    device_vocab = self._get_unique_device_vocab(sdf)
                    device_vocab.save_vocab(args.device_vocab_path)
                    self.device_vocab = device_vocab
                    logger.info(f"Device vocab saved to {args.device_vocab_path}")
            else:
                if self.device_vocab is None:
                    if os.path.exists(args.device_vocab_path):
                        self.device_vocab = DeviceVocab.load_vocab(args.device_vocab_path)
                        logger.info(f"Loaded device vocab from {args.device_vocab_path}")
                    else:
                        raise RuntimeError("Device vocab not found for non-train data")

            if not os.path.exists(temp_dir):
                dates = sorted([row.date for row in sdf.select("date").distinct().collect()])
                logger.info(f"Total unique dates: {len(dates)}")
                dates_json_path = os.path.join(args.output_dir, f"{data_type}_dates.json")
                with open(dates_json_path, 'w') as f:
                    json.dump([d.strftime('%Y-%m-%d') for d in dates], f, indent=2)
                logger.info(f"Saved dates to {dates_json_path}")
            else:
                dates_json_path = os.path.join(args.output_dir, f"{data_type}_dates.json")
                with open(dates_json_path, 'r') as f:
                    dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in json.load(f)]
                logger.info(f"Loaded {len(dates)} dates from {dates_json_path}")

            data_dir = os.path.join(args.output_dir, data_type)
            os.makedirs(data_dir, exist_ok=True)

            # Check max existing processed date
            max_existing_date = None
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    if f.startswith(f"{data_type}_") and f.endswith("_sequences.pkl"):
                        try:
                            date_str = f.replace(f"{data_type}_", "").replace("_sequences.pkl", "")
                            file_date = datetime.strptime(date_str, "%Y%m%d").date()
                            if max_existing_date is None or file_date > max_existing_date:
                                max_existing_date = file_date
                        except ValueError:
                            continue

            if max_existing_date:
                dates_to_process = [d for d in dates if d > max_existing_date]
                logger.info(f"Found max existing date: {max_existing_date}")
                logger.info(f"Processing {len(dates_to_process)} dates after {max_existing_date}")
            else:
                dates_to_process = dates
                logger.info(f"No existing data found, processing all {len(dates_to_process)} dates")
            
            if not dates_to_process:
                logger.info(f"All dates already processed for {data_type}!")
                return

            # Write partitioned parquet to temp
            if not os.path.exists(temp_dir):
                logger.info("Writing partitioned data to temp directory...")
                sdf_to_process = sdf.filter(F.col("date").isin(dates_to_process))
                sdf_to_process.repartition("date").write.partitionBy("date").mode("overwrite").parquet(temp_dir)

            # Process 2 days at a time
            batch_size = 2
            total_sequences = 0
            
            for i in range(0, len(dates_to_process), batch_size):
                batch_dates = dates_to_process[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_dates)} dates")
                
                batch_folders = []
                for date in batch_dates:
                    date_folder = os.path.join(temp_dir, f"date={date}")
                    if os.path.exists(date_folder):
                        batch_folders.append(date_folder)
                    else:
                        logger.warning(f"No data for date {date}")
                
                if not batch_folders:
                    continue
                
                combined_pdf = spark.read.parquet(*batch_folders).toPandas()
                spark.catalog.clearCache()
                
                seqs, _ = generate_sequences_mp(combined_pdf, args.window_size, args.data_mode, device_vocab=self.device_vocab, vocab=self.vocab)
                del combined_pdf
                logger.info(f"Sample seq for batch: {seqs[0]}")
                
                # Save with last date of the batch
                last_date = batch_dates[-1] if len(batch_dates) == len(batch_folders) else \
                            datetime.strptime(os.path.basename(batch_folders[-1]).replace("date=", ""), "%Y-%m-%d").date()
                date_str = last_date.strftime("%Y%m%d")
                seq_path = os.path.join(data_dir, f"{data_type}_{date_str}_sequences.pkl")
                with open(seq_path, "wb") as file:
                    pickle.dump(seqs, file)
                
                logger.info(f"Batch ending {last_date}: {len(seqs):,} sequences -> {seq_path}")
                total_sequences += len(seqs)
                del seqs

            logger.info(f"Processed {len(dates_to_process)} dates with {total_sequences:,.0f} sequences\n")

        finally:
            spark.stop()

    def run(self, exclude: list = []):
        list_prep = ["test", "train"]
        if exclude:
            list_prep = [i for i in list_prep if i not in exclude]

        base = r"/dungnq/DeviceIncidents/data/sample_data/final/remapped"
        for t in list_prep:
            data_path = os.path.join(base, f"{t}.parquet")
            self._process_data(data_path=data_path)
        logger.info("FINISHED PROCESSING ALL DATA TO PICKLE")
