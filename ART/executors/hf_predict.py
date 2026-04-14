import os
import argparse
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import pytz
vntz = pytz.timezone("Asia/Ho_Chi_Minh")

from model_architecture.vocab_builder import WordVocab, DeviceVocab
from model_architecture.helper_utils import visualize_roc_auc, return_percentile_gain_chart, plot_cm, multi_load_pkl
from model_architecture.hf import LogBertForSequenceClassification, DataCollatorWithLogDataset, LogBertConfig
from model_architecture.preprocessing.sampler import BucketBatchSampler
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)


class SequenceTupleDataset(Dataset):
    def __init__(self, seqs: List[Tuple]):
        self.seqs = seqs
        # Alias to support BucketBatchSampler that reads dataset.log_corpus
        self.log_corpus = self.seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx]


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="HF model dir saved by save_pretrained")
    p.add_argument("--test_dir", type=str, required=True, help="Folder of test PKL parts")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--vocab_path", type=str, default=None, help="If None, will try model_dir/vocab.pkl")
    p.add_argument("--device_vocab_path", type=str, default=None, help="If None, will try model_dir/device_vocab.pkl")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=None)
    return p


def _log_error(msg: str):
    try:
        root = Path(__file__).resolve().parents[1]
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "error.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def predict_with_args(args):
    try:
        os.makedirs(args.output_dir, exist_ok=True)

        vocab_path = args.vocab_path or os.path.join(args.model_dir, "vocab.pkl")
        device_vocab_path = args.device_vocab_path or os.path.join(args.model_dir, "device_vocab.pkl")

        vocab: WordVocab = WordVocab.load_vocab(vocab_path)
        device_vocab: DeviceVocab = DeviceVocab.load_vocab(device_vocab_path)
        logger.info(f"Vocab size: {len(vocab)}, dev_vocab size: {len(device_vocab)}")
        model = None
        if args.model_dir == 'init':
            config = LogBertConfig(
                vocab_size=8120,
                hidden_size=256,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=512,
                max_position_embeddings=36000,
                is_time=False,
                is_device=False,
                num_devices=1215,
                use_mlm=True,
                causal=True,
                pad_token_id=0,
            )
            model = LogBertForSequenceClassification(config)  
            logger.info("##Init model without pretrain##")
        else:
            model = LogBertForSequenceClassification.from_pretrained(args.model_dir)
            logger.info(f"LOADED MODEL FROM {args.model_dir}")

       

        logger.info(f"Model architecture:\n{model}")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        sequences = multi_load_pkl(args.test_dir)
        logger.info(f"Loaded seq from {args.test_dir}")

        # agg all data for inference
        # sequences_train = multi_load_pkl(args.train_dir)
        # sequences_val = multi_load_pkl(args.valid_dir)
        # sequences = sequences_train + sequences_val + sequences

        ds = SequenceTupleDataset(sequences)
        collator = DataCollatorWithLogDataset(vocab=vocab, seq_len=args.seq_len, use_mlm=False, mask_ratio=0.0, predict_mode=True)
        batch_sampler = BucketBatchSampler(ds, batch_size=args.batch_size, drop_last=False, seed=args.seed, mode="eval")
        loader = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=args.num_workers if hasattr(args, "num_workers") else int(np.ceil(os.cpu_count() / 2)),
            pin_memory=torch.cuda.is_available(),
        )

        tp = fp = tn = fn = 0
        preds_all: List[int] = []
        gts_all: List[int] = []
        probs_all: List[float] = []

        pred_records = []
        num_batch = 0.0
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader)):
                input_ids = batch["input_ids"].to(device)
                device_ids = batch["device_ids"].to(device)
                labels = batch["labels"].to(device)

                out = model(input_ids=input_ids, device_ids=device_ids, labels=labels)
                logits = out.logits
                soft = torch.softmax(logits, dim=-1)
                probs = soft[:, 1]
                pred = torch.argmax(soft, dim=-1)

                valid = labels >= 0
                if valid.sum().item() < labels.size(0):
                    logger.info(f"Valid samples in batch: {valid.sum().item()}/{labels.size(0)}")
                pv, gv = pred[valid], labels[valid]
                tp += int(((pv == 1) & (gv == 1)).sum().item())
                fp += int(((pv == 1) & (gv == 0)).sum().item())
                tn += int(((pv == 0) & (gv == 0)).sum().item())
                fn += int(((pv == 0) & (gv == 1)).sum().item())

                preds_all.extend(pv.detach().int().cpu().tolist())
                gts_all.extend(gv.detach().int().cpu().tolist())
                probs_all.extend(probs[valid].detach().cpu().tolist())

                total_loss += out.loss_cls
                num_batch += 1


                # append index and all data
                device_ids = batch["device_ids"].detach().cpu().tolist()
                timestamps = batch["window_5min_end"].detach().cpu().tolist()

                pred_records.extend([{
                    "ip": device_vocab.itos[device_id[0]] if len(device_id) > 0 else "unknown",
                    "timestamp": pd.to_datetime(timestamp, unit='s', utc=True).tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m-%d %H:%M:%S"),
                    "pred_proba": float(prob),
                    "labels": int(l),
                } for device_id, timestamp, prob, l in zip(device_ids, timestamps, probs.detach().cpu().tolist(), labels.detach().cpu().tolist())])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        logger.info(f"TP={tp} FP={fp} TN={tn} FN={fn} | P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

        logger.info(f"cls loss: {total_loss/num_batch}")

        # save artifacts
        df = pd.DataFrame(
            pred_records,
            columns=["ip", "timestamp", "pred_proba", "labels"],
        )
        df.to_parquet(os.path.join(args.output_dir, "hf_logbert_predictions.parquet"), index=False)

        df_test = pd.DataFrame({"pred": preds_all, "gt": gts_all, "prob": probs_all})

        visualize_roc_auc(np.array(gts_all), np.array(probs_all), save_fig=True, output_dir=os.path.join(args.output_dir, "roc_auc"))
        return_percentile_gain_chart(df_test, true_col="gt", y_pred="pred", y_proba="prob", number_of_thresholds=10, save_fig=True, output_dir=os.path.join(args.output_dir, "gain_chart"), plot_name="(HF LogBert) Precision-Coverage by Decile")
        plot_cm(df_test["gt"], df_test["pred"], out=os.path.join(args.output_dir, "conf_matrix", f"pred_cm_{datetime.now(vntz).strftime('%d-%m-%y_%H-%M')}.png"))
    except Exception as e:
        _log_error(f"[hf_predict] {type(e).__name__}: {e}")
        raise


def main():
    args = build_argparser().parse_args()
    predict_with_args(args)


if __name__ == "__main__":
    main()