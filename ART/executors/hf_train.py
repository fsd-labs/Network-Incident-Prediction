import os
import argparse
import pickle
import math
from pathlib import Path
from typing import List, Tuple
import gc
import torch.distributed as dist

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, set_seed, TrainerCallback

from model_architecture.vocab_builder import WordVocab, DeviceVocab
from model_architecture.helper_utils import multi_load_pkl
from model_architecture.hf import LogBertConfig, LogBertForSequenceClassification, DataCollatorWithLogDataset
from model_architecture.preprocessing.sampler import BucketBatchSampler
import warnings
warnings.filterwarnings("ignore")
import logging
from model_architecture.hf.hf_utils import BucketTrainer, EarlyStopCallback, CudaCacheClearCallback, InitialLossCallback
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger = logging.getLogger(__name__)

class SequenceTupleDataset(Dataset):
    def __init__(self, seqs: List[Tuple]):
        self.seqs = seqs
        # Alias for compatibility with BucketBatchSampler expecting dataset.log_corpus
        self.log_corpus = self.seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx]


def calc_class_weight(labels: List[int]) -> torch.Tensor:
    if len(labels) == 0:
        return torch.ones(2, dtype=torch.float)
    counts = np.bincount(np.array(labels), minlength=2)
    if counts.sum() == 0:
        return torch.ones(2, dtype=torch.float)
    n_samples = float(counts.sum())
    n_classes = float(len(counts))
    weights = n_samples / (n_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, required=True, help="Folder of train PKL parts")
    p.add_argument("--valid_dir", type=str, default=None, help="Folder of valid PKL parts")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)
    p.add_argument("--device_vocab_path", type=str, required=True)

    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--attn_heads", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=12288)
    p.add_argument("--is_device", action="store_true", default=False)
    p.add_argument("--use_mlm", action="store_true", default=False)
    p.add_argument("--mask_ratio", type=float, default=0.0)
    p.add_argument("--causal", action="store_true", default=False)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="linear",
                   choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                   help="Learning rate scheduler type")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--fsdp", type=str, default=None)
    p.add_argument("--fsdp_config", type=str, default=None)
    p.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="TransformerBlock")

    # Saving/eval strategy
    p.add_argument("--save_strategy", type=str, choices=["steps", "epoch"], default="epoch")
    p.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default="steps")
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--cuda_clear_each_epoch", action="store_true", default=False,
                   help="If set, clear CUDA cache and run gc.collect() at each epoch end")
    p.add_argument("--log_loss_components", action="store_true", default=False,
                   help="Log loss_cls/loss_mlm and MLM nonzero ratio for debugging")

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

def _get_dist():
    if "WORLD_SIZE" in os.environ:
        try:
            ws = int(os.environ["WORLD_SIZE"])
            rk = int(os.environ.get("RANK", 0))
            logger.info("AVAILABLE WORLD SIZE VARIABLES")
            return ws, rk
        except ValueError:
            pass

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    return 1, 0

def train_with_args(args):
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        set_seed(args.seed)
        
        # Get distributed info EARLY
        world_size, rank = _get_dist()
        is_main = rank == 0
        
        if is_main:
            logger.info(f"Output dir: {args.output_dir}")
            logger.info(f"Seed set: {args.seed}")

        vocab: WordVocab = WordVocab.load_vocab(args.vocab_path)
        device_vocab: DeviceVocab = DeviceVocab.load_vocab(args.device_vocab_path)
        if is_main:
            logger.info(f"VOCAB UNK INDEX: {vocab.unk_index}")
            logger.info(f"Vocab size: {len(vocab)}")
            logger.info(f"Device Vocab size: {len(device_vocab)}")

        if is_main:
            logger.info("Loading training data...")
        train_sequences = multi_load_pkl(args.train_dir)
        if args.valid_dir and os.path.exists(args.valid_dir):
            valid_sequences = multi_load_pkl(args.valid_dir)
        else:
            valid_sequences = None
        if is_main:
            logger.info(f"Train sequences: {len(train_sequences)}")
            logger.info(f"Valid sequences: {0 if valid_sequences is None else len(valid_sequences)}")

        if len(train_sequences) > 0:
            if getattr(args, "fix_seq_len", False):
                if is_main:
                    logger.info(f"Using fixed seq_len: {args.seq_len}")
            else:
                max_len_train = max(len(x[0]) for x in train_sequences) + 1
                args.seq_len = max(args.seq_len, max_len_train)
                if is_main:
                    logger.info(f"Seq len set to {args.seq_len} (max from train {max_len_train})")

        train_ds = SequenceTupleDataset(train_sequences)
        eval_ds = SequenceTupleDataset(valid_sequences) if valid_sequences else None
        if is_main:
            try:
                logger.info(f"Dataset sizes - train: {len(train_ds):,.0f} eval: {0 if eval_ds is None else len(eval_ds)}")
            except Exception:
                pass

        use_mlm_flag = getattr(args, "use_mlm", getattr(args, "is_logkey", False))

        config = LogBertConfig(
            vocab_size=len(vocab),
            hidden_size=args.hidden,
            num_hidden_layers=args.layers,
            num_attention_heads=args.attn_heads,
            intermediate_size=args.hidden * 2,
            max_position_embeddings=args.seq_len,
            is_time=False,
            is_device=args.is_device,
            num_devices=len(device_vocab),
            use_mlm=use_mlm_flag,
            num_labels=2,
            causal=args.causal,
            use_l1=args.l1_norm,
        )
        logger.info(f"Using l1_norm: {args.l1_norm}")

        from_pretrained_weights = getattr(args, "model_weights_path", None)
        if from_pretrained_weights and os.path.exists(from_pretrained_weights):
            if is_main:
                logger.info(f"Loading model weights from {from_pretrained_weights}")
            model = LogBertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=from_pretrained_weights,
                config=config,
            )
            args.ckpt_subfolder = f"{args.ckpt_subfolder}_from_weights"
            args.tf_logs_subfolder = f"{args.tf_logs_subfolder}_from_weights"
            if is_main:
                logger.info(f"Modified ckpt and tf_log subfolder to {args.ckpt_subfolder} and {args.tf_logs_subfolder}")
        else:
            if is_main:
                logger.info("Initializing model from scratch")
            model = LogBertForSequenceClassification(config)
        

        y_train = [int(t[3]) if len(t) > 3 else -1 for t in train_sequences]
        y_train = [y for y in y_train if y >= 0]
        cw = calc_class_weight(y_train)
        if is_main:
            logger.info(f"Class weights: {cw}")
        
        # Set class weight on CPU - it will be moved to correct device by DDP
        model.set_class_weight(cw)

        logger.info("\n" + "="*80)
        logger.info("MODEL ARCHITECTURE:")
        logger.info("="*80)
        logger.info(f"\n{model}")

        # Đếm số parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("\n" + "="*80)
        logger.info("MODEL PARAMETERS:")
        logger.info("="*80)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        logger.info("="*80 + "\n")

        data_collator = DataCollatorWithLogDataset(
            vocab=vocab,
            seq_len=args.seq_len,
            use_mlm=use_mlm_flag,
            mask_ratio=args.mask_ratio,
            predict_mode=False,
        )

        # multi-gpu setup
        if world_size > 1:
            logger.info(f"Distributed training setup detected: world_size={world_size}, rank={rank}")
            # new lr
            new_lr = args.lr * world_size * 0.68 if getattr(args, "multi_gpu_lr_scale", False) else args.lr
            logger.info(f"Adjusting learning rate for multi-gpu: {args.lr} --> {new_lr}")
            args.lr = new_lr
            logger.info(f"VALID CHECK LR CHANGE: {args.lr == new_lr}")
            # new warmup steps
            new_warmup = int(np.ceil(args.warmup_steps / world_size))
            logger.info(f"Adjusting warmup steps for multi-gpu: {args.warmup_steps} --> {new_warmup}")
            args.warmup_steps = new_warmup
            logger.info(f"VALID CHECK WARMUP CHANGE: {args.warmup_steps == new_warmup}")
        else:
            logger.info("Single GPU or CPU training setup detected")
        logger.info(f"Schedule: {args.lr_scheduler_type}")

        eff_bs = args.batch_size * max(1, args.gradient_accumulation_steps) * max(1, world_size)
        steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bs))

        # scale total
        total_training_steps = steps_per_epoch * args.epochs
        warmup_rate = 0.1
        updated_warmup_steps = int(total_training_steps * warmup_rate)
        
        if args.warmup_steps < updated_warmup_steps:
            if is_main:
                logger.info(f"Dynamically adjusting warmup steps: {args.warmup_steps} --> {updated_warmup_steps}")
            args.warmup_steps = updated_warmup_steps

        wants_epoch_save = (args.save_strategy == "epoch")
        logger.info(f"Effective batch size: {eff_bs}, steps/epoch: {steps_per_epoch}, save_strategy={args.save_strategy}")

        # Eval scheduling: if user sets evaluation_strategy=epoch, run every K epochs (default 3)
        eval_steps_final = None
        if eval_ds is not None:
            if getattr(args, "evaluation_strategy", "steps") == "epoch":
                k_epochs = max(1, int(getattr(args, "eval_steps", 3)))
                eval_steps_final = k_epochs * steps_per_epoch
            else:
                eval_steps_final = max(1, int(getattr(args, "eval_steps", 500)))
        if eval_ds is not None:
            logger.info(
                f"Evaluation scheduling resolved to eval_steps={eval_steps_final} (strategy={getattr(args, 'evaluation_strategy', 'steps')})"
            )

        # Build TrainingArguments
        fsdp_value = getattr(args, "fsdp", None)
        fsdp_config_value = None
        if fsdp_value:
            fsdp_config_arg = getattr(args, "fsdp_config", None)
            if fsdp_config_arg:
                fsdp_config_value = fsdp_config_arg
            else:
                fsdp_transformer_cls = getattr(
                    args, "fsdp_transformer_layer_cls_to_wrap", "TransformerBlock"
                )
                fsdp_config_value = {
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_transformer_layer_cls_to_wrap": fsdp_transformer_cls,
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                    "fsdp_use_orig_params": True,
                    "fsdp_sync_module_states": True,
                    "fsdp_cpu_ram_efficient_loading": True,
                    "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
                    "fsdp_forward_prefetch": False,
                    "fsdp_offload_params": False,
                    "activation_checkpointing": False,
                    "limit_all_gathers": False,
                }

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, args.ckpt_subfolder),
            logging_dir=os.path.join(args.output_dir, args.tf_logs_subfolder),
            # overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_strategy=args.logging_strategy,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=(steps_per_epoch if wants_epoch_save else args.save_steps),
            eval_strategy=("steps" if eval_ds is not None else "no"),
            eval_steps=(eval_steps_final if eval_ds is not None and eval_steps_final is not None else None),
            save_total_limit=args.save_total_limit,
            # load_best_model_at_end=True, --require save_steps = bội của eval_steps
            fp16=args.fp16,
            bf16=args.bf16,
            dataloader_num_workers=args.num_workers,
            report_to=["tensorboard"],
            eval_accumulation_steps=100,
            prediction_loss_only=True,
            ddp_find_unused_parameters=False,
            fsdp=fsdp_value,
            fsdp_config=fsdp_config_value,
        )
        logger.info(
            f"Training args: epochs={training_args.num_train_epochs}, lr={training_args.learning_rate}, "
            f"logging_strategy={training_args.logging_strategy}, save_steps={training_args.save_steps}, eval_steps={training_args.eval_steps}"
        )

        if args.continue_train:
            resume_path = args.resume_from.rstrip("/")
            last_ckpt = int(os.path.basename(resume_path).split("-")[1])
            args.epoch_resume = int(np.ceil(last_ckpt / training_args.save_steps))
        else:
            args.epoch_resume=0
        logger.info(f"SET SAMPLER SEED TO EPOCH {args.epoch_resume}")

        trainer = BucketTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            # compute_metrics=compute_metrics if eval_ds is not None else None,
            compute_metrics=None,
            resume_from_epoch=args.epoch_resume,
            log_loss_components=getattr(args, "log_loss_components", False),
        )

        # Log initial loss at step 0 (before training starts)
        if not args.continue_train:
            try:
                trainer.add_callback(InitialLossCallback(
                    trainer=trainer,
                    log_initial_train=True,
                    log_initial_eval=(eval_ds is not None)
                ))
                logger.info("Added InitialLossCallback to log loss at step 0")
            except Exception as e:
                logger.warning(f"Failed to add InitialLossCallback: {e}")

        try:
            logger.info(
                f"Enable EarlyStopping: patience={getattr(args, 'early_stop', 4)}, min_delta={getattr(args, 'min_delta', 0.0)}"
            )
            trainer.add_callback(EarlyStopCallback(patience=getattr(args, "early_stop", 4), min_delta=getattr(args, "min_delta", 0.0), ini_best=getattr(args, "ini_best_es", None)))
        except Exception as e:
            logger.warning(f"Failed to add EarlyStopCallback: {e}")
            pass

        # Optional CUDA cache clearing per epoch
        if getattr(args, "cuda_clear_each_epoch", False):
            try:
                logger.info("Enable CUDA cache clearing at each epoch end")
                trainer.add_callback(CudaCacheClearCallback())
            except Exception as e:
                logger.warning(f"Failed to add CudaCacheClearCallback: {e}")
                pass

        # Resume from checkpoint
        resume_ckpt = getattr(args, "resume_from", None) if getattr(args, "continue_train", False) else None
        if resume_ckpt:
            logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
        else:
            logger.info("Starting training from scratch")
        trainer.train(resume_from_checkpoint=resume_ckpt)

        save_dir = os.path.join(args.output_dir, "hf_model" + (f"_from_weights" if from_pretrained_weights else ""))
        logger.info(f"Saving model to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        trainer.model.save_pretrained(save_dir, safe_serialization=True)
        config.save_pretrained(save_dir)

        # save artifacts
        with open(os.path.join(save_dir, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)
        with open(os.path.join(save_dir, "device_vocab.pkl"), "wb") as f:
            pickle.dump(device_vocab, f)

        print(f"Saved HF model to {save_dir}")
        try:
            logger.info(f"Saved HF model to {save_dir}")
            logger.info("Training finished successfully")
        except Exception:
            pass
    except Exception as e:
        _log_error(f"[hf_train] {type(e).__name__}: {e}")
        raise


def main():
    args = build_argparser().parse_args()
    train_with_args(args)


if __name__ == "__main__":
    main()
