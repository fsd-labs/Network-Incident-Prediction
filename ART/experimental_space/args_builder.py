from argparse import Namespace
import argparse
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
# artifacts/

DEFAULTS: Dict[str, Any] = {
    # Basic
    "mode": "train",
    "train_parquet": "/home/minhnd74/DeviceIncidents/data/sample_data/final/remapped/train.parquet",
    "valid_parquet": "/home/minhnd74/DeviceIncidents/data/sample_data/final/remapped/valid.parquet",
    "predict_parquet": "/home/minhnd74/DeviceIncidents/data/sample_data/final/remapped/test.parquet",
    "ckpt_subfolder": "hf_runs",
    "tf_logs_subfolder": "tf_logs",

    "output_dir": "artifacts/",
    "model_dir": "artifacts/hf_runs/checkpoint-5976",
    # "model_dir": "artifacts/hf_model",
    "vocab_path": "artifacts/vocab.pkl",
    "device_vocab_path": "artifacts/dev_vocab.pkl",
    "model_weights_path": None,

    # Optional PKL dirs (when using HF executors)
    "train_dir": "artifacts/train/", # sample data to test
    "valid_dir": "artifacts/valid/",
    "test_dir": "artifacts/test_org/",

    # Data/sequence
    "min_freq": 1,
    "window_size": 5,
    "data_mode": "time_window",  # "fixed_size" or "time_window"
    "seq_len": 36000,
    "fix_seq_len": True,
    "mask_ratio": 0.0,
    "is_logkey": True,
    "is_device": False,
    "causal": True,
    "l1_norm": True,

    # Train
    "epochs": 20,
    "batch_size": 8,
    "num_workers": os.cpu_count() if os.cpu_count() is not None else 4,
    # "num_workers": 4,
    "lr": 5e-5,
    "betas": [0.9, 0.999],
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "lr_scheduler_type": "linear",
    "log_freq": 100,
    "logging_strategy": "epoch",
    "logging_steps": 1000,
    "early_stop": 4,
    "min_delta": 0.002,
    "continue_train": True,
    "resume_from": "artifacts/hf_runs/checkpoint-15750",
    "alpha_mlm": 0.1,
    "accum_steps": 32,
    "gradient_accumulation_steps": 32,
    "multi_gpu_lr_scale": True,

    # HF Trainer strategies
    "seed": 42,
    "save_strategy": "epoch",  # "steps" or "epoch"
    "evaluation_strategy": "epoch",  # "no", "steps", "epoch"
    "save_total_limit": 50,
    "save_steps": 500,
    "eval_steps": 1,
    "fp16": False,
    "bf16": True,
    "cuda_clear_each_epoch": True,
    # "fsdp": "full_shard auto_wrap",
    "fsdp": None,
    "fsdp_config": None,
    "fsdp_transformer_layer_cls_to_wrap": "TransformerBlock",
    # "ini_best_es": 0.915867,

    # Model arch
    "hidden": 256,
    "layers": 1,
    "attn_heads": 4,

    # System
    "no_cuda": False,
    "log_loss_components": True,
}


def build_parser(defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    defs = DEFAULTS if defaults is None else {**DEFAULTS, **defaults}
    parser = argparse.ArgumentParser(description="Refactored execution for LogBERT")
    parser.add_argument("--mode", type=str, choices=["vocab", "train", "predict"], default=defs["mode"])
    parser.add_argument("--train_parquet", type=str, default=defs["train_parquet"])
    parser.add_argument("--valid_parquet", type=str, default=defs["valid_parquet"])
    parser.add_argument("--predict_parquet", type=str, default=defs["predict_parquet"])
    parser.add_argument("--output_dir", type=str, default=defs["output_dir"])
    parser.add_argument("--model_dir", type=str, default=defs["model_dir"])
    parser.add_argument("--vocab_path", type=str, default=defs["vocab_path"])
    parser.add_argument("--device_vocab_path", type=str, default=defs["device_vocab_path"])
    parser.add_argument("--model_weights_path", type=str, default=defs["model_weights_path"])
    # Optional PKL dirs for HF executors
    parser.add_argument("--train_dir", type=str, default=defs["train_dir"])
    parser.add_argument("--valid_dir", type=str, default=defs["valid_dir"])
    parser.add_argument("--test_dir", type=str, default=defs["test_dir"])
    parser.add_argument("--ckpt_subfolder", type=str, default=defs["ckpt_subfolder"])
    parser.add_argument("--tf_logs_subfolder", type=str, default=defs["tf_logs_subfolder"])
    parser.add_argument("--min_freq", type=int, default=defs["min_freq"])
    parser.add_argument("--window_size", type=float, default=defs["window_size"])
    parser.add_argument("--data_mode", type=str, choices=["fixed_size", "time_window"], default=defs["data_mode"])
    parser.add_argument("--seq_len", type=int, default=defs["seq_len"])
    parser.add_argument("--fix_seq_len", action="store_true", default=defs["fix_seq_len"])
    parser.add_argument("--mask_ratio", type=float, default=defs["mask_ratio"])
    parser.add_argument("--is_logkey", action="store_true", default=defs["is_logkey"])
    parser.add_argument("--is_device", action="store_true", default=defs["is_device"])
    parser.add_argument("--causal", action="store_true", default=defs["causal"])

    parser.add_argument("--epochs", type=int, default=defs["epochs"])
    parser.add_argument("--batch_size", type=int, default=defs["batch_size"])
    parser.add_argument("--num_workers", type=int, default=defs["num_workers"])
    parser.add_argument("--lr", type=float, default=defs["lr"])
    parser.add_argument("--betas", type=float, nargs=2, default=defs["betas"])
    parser.add_argument("--weight_decay", type=float, default=defs["weight_decay"])
    parser.add_argument("--warmup_steps", type=int, default=defs["warmup_steps"])
    parser.add_argument("--lr_scheduler_type", type=str,
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        default=defs["lr_scheduler_type"])
    parser.add_argument("--log_freq", type=int, default=defs["log_freq"])
    parser.add_argument("--logging_steps", type=int, default=defs["logging_steps"])
    parser.add_argument("--early_stop", type=int, default=defs["early_stop"])
    parser.add_argument("--min_delta", type=float, default=defs["min_delta"])
    parser.add_argument("--continue_train", action="store_true", default=defs["continue_train"])
    parser.add_argument("--resume_from", type=str, default=defs["resume_from"])
    parser.add_argument("--alpha_mlm", type=float, default=defs["alpha_mlm"])
    parser.add_argument("--accum_steps", type=int, default=defs["accum_steps"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=defs["gradient_accumulation_steps"])
    parser.add_argument("--multi_gpu_lr_scale", action="store_true", default=defs["multi_gpu_lr_scale"])
    # HF Trainer strategies
    parser.add_argument("--seed", type=int, default=defs["seed"])
    parser.add_argument("--save_strategy", type=str, choices=["steps", "epoch"], default=defs["save_strategy"])
    parser.add_argument("--logging_strategy", type=str, choices=["steps", "epoch"], default=defs["logging_strategy"])
    parser.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default=defs["evaluation_strategy"])
    parser.add_argument("--save_total_limit", type=int, default=defs["save_total_limit"])
    parser.add_argument("--save_steps", type=int, default=defs["save_steps"])
    parser.add_argument("--eval_steps", type=int, default=defs["eval_steps"])
    parser.add_argument("--fp16", action="store_true", default=defs["fp16"])
    parser.add_argument("--bf16", action="store_true", default=defs["bf16"])
    parser.add_argument("--cuda_clear_each_epoch", action="store_true", default=defs["cuda_clear_each_epoch"])
    parser.add_argument("--fsdp", type=str, default=defs["fsdp"])
    parser.add_argument("--fsdp_config", type=str, default=defs["fsdp_config"])
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        default=defs["fsdp_transformer_layer_cls_to_wrap"],
    )
    parser.add_argument("--hidden", type=int, default=defs["hidden"])
    parser.add_argument("--layers", type=int, default=defs["layers"])
    parser.add_argument("--attn_heads", type=int, default=defs["attn_heads"])
    parser.add_argument("--no_cuda", action="store_true", default=defs["no_cuda"])
    parser.add_argument("--log_loss_components", action="store_true", default=defs["log_loss_components"],
                   help="Log loss_cls/loss_mlm and MLM nonzero ratio for debugging")
    parser.add_argument("--l1_norm", action="store_true", default=defs["l1_norm"])


    return parser


def get_args(overrides: Optional[Dict[str, Any]] = None) -> Namespace:
    defs = DEFAULTS if overrides is None else {**DEFAULTS, **overrides}
    return Namespace(**defs)

args = get_args()

__all__ = ["DEFAULTS", "build_parser", "get_args", "args"]
