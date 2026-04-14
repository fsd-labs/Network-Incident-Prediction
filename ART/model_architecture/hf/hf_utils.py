import gc
import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from model_architecture.preprocessing.sampler import BucketBatchSampler

logger = logging.getLogger(__name__)


class BucketTrainer(Trainer):
    """
    Custom Trainer with bucket-based batch sampling for variable-length sequences.

    """
    def __init__(self, *args, resume_from_epoch=None, log_loss_components: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_from_epoch = resume_from_epoch
        self.log_loss_components = log_loss_components
        self._reset_loss_component_buffers()

    def _reset_loss_component_buffers(self):
        self._train_loss_cls_sum = 0.0
        self._train_loss_cls_count = 0
        self._train_loss_mlm_sum = 0.0
        self._train_loss_mlm_count = 0
        self._train_loss_l1_sum = 0.0
        self._train_loss_l1_count = 0
        self._train_mlm_ratio_sum = 0.0
        self._train_mlm_ratio_count = 0
        self._eval_loss_cls_sum = 0.0
        self._eval_loss_cls_count = 0
        self._eval_loss_mlm_sum = 0.0
        self._eval_loss_mlm_count = 0
        self._eval_loss_l1_sum = 0.0
        self._eval_loss_l1_count = 0
        self._eval_mlm_ratio_sum = 0.0
        self._eval_mlm_ratio_count = 0

    def _reset_train_loss_buffers(self):
        self._train_loss_cls_sum = 0.0
        self._train_loss_cls_count = 0
        self._train_loss_mlm_sum = 0.0
        self._train_loss_mlm_count = 0
        self._train_loss_l1_sum = 0.0
        self._train_loss_l1_count = 0
        self._train_mlm_ratio_sum = 0.0
        self._train_mlm_ratio_count = 0

    def _reset_eval_loss_buffers(self):
        self._eval_loss_cls_sum = 0.0
        self._eval_loss_cls_count = 0
        self._eval_loss_mlm_sum = 0.0
        self._eval_loss_mlm_count = 0
        self._eval_loss_l1_sum = 0.0
        self._eval_loss_l1_count = 0
        self._eval_mlm_ratio_sum = 0.0
        self._eval_mlm_ratio_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if self.log_loss_components:
            loss_cls = getattr(outputs, "loss_cls", None)
            loss_mlm = getattr(outputs, "loss_mlm", None)
            loss_l1 = getattr(outputs, "loss_l1", None)
            if loss_cls is None and isinstance(outputs, dict):
                loss_cls = outputs.get("loss_cls")
            if loss_mlm is None and isinstance(outputs, dict):
                loss_mlm = outputs.get("loss_mlm")
            if loss_l1 is None and isinstance(outputs, dict):
                loss_l1 = outputs.get("loss_l1")
            loss_cls_val = None
            loss_mlm_val = None
            loss_l1_val = None
            if loss_cls is not None:
                try:
                    loss_cls_val = loss_cls.detach().float().mean().item()
                except Exception:
                    loss_cls_val = None
            if loss_mlm is not None:
                try:
                    loss_mlm_val = loss_mlm.detach().float().mean().item()
                except Exception:
                    loss_mlm_val = None
            if loss_l1 is not None:
                try:
                    loss_l1_val = loss_l1.detach().float().mean().item()
                except Exception:
                    loss_l1_val = None

            mlm_labels = inputs.get("mlm_labels")
            ignore_index = None
            if hasattr(model, "mlm_loss_fn"):
                ignore_index = getattr(model.mlm_loss_fn, "ignore_index", None)
            if ignore_index is None:
                ignore_index = 0
            mlm_ratio = None
            if mlm_labels is not None:
                try:
                    mlm_ratio = (mlm_labels != ignore_index).float().mean().item()
                except Exception:
                    mlm_ratio = None

            if model.training:
                if loss_cls_val is not None:
                    self._train_loss_cls_sum += loss_cls_val
                    self._train_loss_cls_count += 1
                if loss_mlm_val is not None:
                    self._train_loss_mlm_sum += loss_mlm_val
                    self._train_loss_mlm_count += 1
                if loss_l1_val is not None:
                    self._train_loss_l1_sum += loss_l1_val
                    self._train_loss_l1_count += 1
                if mlm_ratio is not None:
                    self._train_mlm_ratio_sum += mlm_ratio
                    self._train_mlm_ratio_count += 1
            else:
                if loss_cls_val is not None:
                    self._eval_loss_cls_sum += loss_cls_val
                    self._eval_loss_cls_count += 1
                if loss_mlm_val is not None:
                    self._eval_loss_mlm_sum += loss_mlm_val
                    self._eval_loss_mlm_count += 1
                if loss_l1_val is not None:
                    self._eval_loss_l1_sum += loss_l1_val
                    self._eval_loss_l1_count += 1
                if mlm_ratio is not None:
                    self._eval_mlm_ratio_sum += mlm_ratio
                    self._eval_mlm_ratio_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self.log_loss_components and logs and "loss" in logs:
            extra = {}
            if self._train_loss_cls_count > 0:
                extra["train_loss_cls"] = self._train_loss_cls_sum / self._train_loss_cls_count
            if self._train_loss_mlm_count > 0:
                extra["train_loss_mlm"] = self._train_loss_mlm_sum / self._train_loss_mlm_count
            if self._train_loss_l1_count > 0:
                extra["train_loss_l1"] = self._train_loss_l1_sum / self._train_loss_l1_count
            if self._train_mlm_ratio_count > 0:
                extra["train_mlm_nonzero_ratio"] = self._train_mlm_ratio_sum / self._train_mlm_ratio_count
            if extra:
                logs = dict(logs)
                logs.update(extra)
                self._reset_train_loss_buffers()
        return super().log(logs, *args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.log_loss_components:
            self._reset_eval_loss_buffers()
        metrics = super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        if self.log_loss_components and self.is_world_process_zero():
            extra_metrics = {}
            if self._eval_loss_cls_count > 0:
                extra_metrics[f"{metric_key_prefix}_loss_cls"] = (
                    self._eval_loss_cls_sum / self._eval_loss_cls_count
                )
            if self._eval_loss_mlm_count > 0:
                extra_metrics[f"{metric_key_prefix}_loss_mlm"] = (
                    self._eval_loss_mlm_sum / self._eval_loss_mlm_count
                )
            if self._eval_loss_l1_count > 0:
                extra_metrics[f"{metric_key_prefix}_loss_l1"] = (
                    self._eval_loss_l1_sum / self._eval_loss_l1_count
                )
            if self._eval_mlm_ratio_count > 0:
                extra_metrics[f"{metric_key_prefix}_mlm_nonzero_ratio"] = (
                    self._eval_mlm_ratio_sum / self._eval_mlm_ratio_count
                )
            if extra_metrics:
                self.log(extra_metrics)
                metrics.update(extra_metrics)
        return metrics

    def _sync_if_distributed(self):
        """Synchronize all processes to prevent race conditions."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def get_train_dataloader(self):
        ds = self.train_dataset
        collate_fn = self.data_collator
        bs = self.args.per_device_train_batch_size
        batch_sampler = BucketBatchSampler(
            ds,
            batch_size=bs,
            drop_last=False,
            seed=getattr(self.args, "seed", 42),
            mode="train",
            resume_from_epoch=self.resume_from_epoch
        )
        # Sync before creating dataloader to ensure all ranks are aligned
        self._sync_if_distributed()
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=getattr(self.args, "dataloader_num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_ds is None:
            return super().get_eval_dataloader(eval_ds)
        collate_fn = self.data_collator
        bs = self.args.per_device_eval_batch_size
        batch_sampler = BucketBatchSampler(
            eval_ds,
            batch_size=bs,
            drop_last=False,
            seed=getattr(self.args, "seed", 42),
            mode = "eval",
            resume_from_epoch=self.resume_from_epoch
        )
        # Sync before creating dataloader
        self._sync_if_distributed()
        return DataLoader(
            eval_ds,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=getattr(self.args, "dataloader_num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )


class EarlyStopCallback(TrainerCallback):
    def __init__(self, patience: int, min_delta: float = 0.0, ini_best: float = None):
        super().__init__()
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.best = ini_best
        self.num_bad = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        if "eval_loss" not in metrics:
            return control
        val = metrics["eval_loss"]
        if val is None or (isinstance(val, float) and (val != val)):
            return control
        if self.best is None:
            self.best = val
            self.num_bad = 0
            logger.info(f"EarlyStop init best={self.best:.6f}")
            return control
        improved = (val < self.best - self.min_delta)
        if improved:
            logger.info(
                f"EarlyStop: improved eval_loss {self.best:.6f} -> {val:.6f} (min_delta={self.min_delta})"
            )
            self.best = val
            self.num_bad = 0
        else:
            self.num_bad += 1
            logger.info(
                f"EarlyStop: no improve eval_loss={val:.6f}, best={self.best:.6f}, bad_epochs={self.num_bad}/{self.patience}"
            )
            if self.num_bad >= self.patience:
                logger.warning("EarlyStop: patience reached, stopping training")
                control.should_training_stop = True
        return control


class CudaCacheClearCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    logger.info(
                        f"CUDA cache cleared. reserved={reserved:.1f}MB allocated={allocated:.1f}MB"
                    )
                except Exception:
                    logger.info("CUDA cache cleared.")
            except Exception:
                pass
        try:
            gc.collect()
        except Exception:
            pass
        return control


class InitialLossCallback(TrainerCallback):
    """
    Callback để tính và log initial loss (loss khi model vừa init, step=0)
    trước khi bắt đầu training loop.
    
    - Với train: model.train() + torch.no_grad() (không backward)
    - Với eval: model.eval() + torch.no_grad() (bình thường)
    """
    
    def __init__(self, trainer=None, log_initial_train: bool = True, log_initial_eval: bool = True):
        super().__init__()
        self.trainer = trainer  # Store trainer reference
        self.log_initial_train = log_initial_train
        self.log_initial_eval = log_initial_eval
        self._logged = False  # Đảm bảo chỉ log 1 lần
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._logged:
            return control
        
        # Get trainer from self.trainer (set during initialization)
        trainer = self.trainer
        if trainer is None:
            logger.warning("InitialLossCallback: trainer reference not available, skipping initial loss")
            return control
        
        logs = {}
        
        # === Log Initial Train Loss ===
        if self.log_initial_train:
            try:
                was_training = model.training
                model.train()  # Training mode
                train_dl = trainer.get_train_dataloader()
                batch = next(iter(train_dl))
                batch = trainer._prepare_inputs(batch)
                
                with torch.no_grad():  # Không tính gradient
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss")
                    if loss is not None:
                        logs["initial_train_loss"] = loss.detach().float().item()
                        
                        # Log thêm loss components nếu có
                        loss_cls = getattr(outputs, "loss_cls", None)
                        loss_mlm = getattr(outputs, "loss_mlm", None)
                        loss_l1 = getattr(outputs, "loss_l1", None)
                        if loss_cls is not None:
                            logs["initial_train_loss_cls"] = loss_cls.detach().float().item()
                        if loss_mlm is not None:
                            logs["initial_train_loss_mlm"] = loss_mlm.detach().float().item()
                        if loss_l1 is not None:
                            logs["initial_train_loss_l1"] = loss_l1.detach().float().item()
                
                # Restore original training state
                if not was_training:
                    model.eval()
            except Exception as e:
                logger.warning(f"Failed to compute initial train loss: {e}")
        
        # === Log Initial Eval Loss ===
        if self.log_initial_eval and trainer.eval_dataset is not None:
            try:
                model.eval()  # Eval mode
                eval_dl = trainer.get_eval_dataloader()
                batch = next(iter(eval_dl))
                batch = trainer._prepare_inputs(batch)
                
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss")
                    if loss is not None:
                        logs["initial_eval_loss"] = loss.detach().float().item()

                        # Log thêm eval loss components nếu có
                        loss_cls = getattr(outputs, "loss_cls", None)
                        loss_mlm = getattr(outputs, "loss_mlm", None)
                        loss_l1 = getattr(outputs, "loss_l1", None)
                        if loss_cls is not None:
                            logs["initial_eval_loss_cls"] = loss_cls.detach().float().item()
                        if loss_mlm is not None:
                            logs["initial_eval_loss_mlm"] = loss_mlm.detach().float().item()
                        if loss_l1 is not None:
                            logs["initial_eval_loss_l1"] = loss_l1.detach().float().item()
            except Exception as e:
                logger.warning(f"Failed to compute initial eval loss: {e}")
        
        # Log với step = 0
        if logs and trainer.is_world_process_zero():
            logger.info(f"Initial losses at step 0: {logs}")
            # Force log at step 0
            old_step = state.global_step
            state.global_step = 0
            trainer.log(logs)
            state.global_step = old_step
        
        self._logged = True
        return control
