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
import matplotlib.pyplot as plt

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

def analyze_linear1_weights(model, logger=None, save_path=None, num_bins=50):
    w = model.classifier.linear1.weight.detach().float().view(-1).cpu()

    w_sum = torch.norm(w, 1)
    if logger:
        logger.info(f"[linear1] sum={w_sum:.6f}")

    # ===== stats =====
    w_min = w.min().item()
    w_max = w.max().item()
    w_mean = w.mean().item()
    w_std = w.std(unbiased=False).item()
    q = torch.quantile(w, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))

    if logger:
        logger.info(f"[linear1] shape={tuple(model.classifier.linear1.weight.shape)}")
        logger.info(f"[linear1] min={w_min:.6f}, max={w_max:.6f}, mean={w_mean:.6f}, std={w_std:.6f}")
        logger.info(
            f"[linear1] q0={q[0].item():.6f}, q25={q[1].item():.6f}, "
            f"q50={q[2].item():.6f}, q75={q[3].item():.6f}, q100={q[4].item():.6f}"
        )

    # ===== plot =====
    fig, ax = plt.subplots(figsize=(8, 5))
    w_np = w.numpy()

    n, bins, patches = ax.hist(w_np, bins=num_bins, density=True, alpha=0.75, color="#4C72B0", edgecolor="none")

    # KDE overlay — đường cong mượt, không can thiệp vào bin
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(w_np, bw_method="scott")
    xs = np.linspace(w_min, w_max, 300)
    ax.plot(xs, kde(xs), color="#C44E52", linewidth=1.6, label="KDE")

    # mean & median
    ax.axvline(w_mean,   color="#DD8452", linewidth=1.2, linestyle="--", label=f"Mean   {w_mean:.4f}")
    ax.axvline(q[2].item(), color="#55A868", linewidth=1.2, linestyle=":",  label=f"Median {q[2].item():.4f}")

    ax.set_title("Weight Distribution — linear1", fontsize=13, pad=10)
    ax.set_xlabel("Weight value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(framealpha=0.5, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        if logger:
            logger.info(f"Saved histogram to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def debug_with_args(args):
    try:
        os.makedirs(args.output_dir, exist_ok=True)

        vocab_path = args.vocab_path or os.path.join(args.model_dir, "vocab.pkl")
        device_vocab_path = args.device_vocab_path or os.path.join(args.model_dir, "device_vocab.pkl")

        vocab: WordVocab = WordVocab.load_vocab(vocab_path)
        device_vocab: DeviceVocab = DeviceVocab.load_vocab(device_vocab_path)
        logger.info(f"Vocab size: {len(vocab)}, dev_vocab size: {len(device_vocab)}")

        if args.model_dir == "init":
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

        hist_path = os.path.join(args.output_dir, f"linear1_hist{args.model_dir}.png")

        analyze_linear1_weights(
            model,
            logger=logger,
            save_path=hist_path,
            num_bins=50
        )

        
    except Exception as e:
        _log_error(f"[hf_predict] {type(e).__name__}: {e}")
        raise


def main():
    args = build_argparser().parse_args()
    debug_with_args(args)


if __name__ == "__main__":
    main()
