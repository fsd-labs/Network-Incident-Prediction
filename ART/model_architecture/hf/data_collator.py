from typing import List, Tuple, Dict, Any
import torch
from ..preprocessing.dataset import LogDataset


class DataCollatorWithLogDataset:
    def __init__(self, vocab, seq_len: int, use_mlm: bool = False, mask_ratio: float = 0.0, predict_mode: bool = False):
        self.vocab = vocab
        self.seq_len = seq_len
        self.use_mlm = use_mlm
        self.mask_ratio = mask_ratio
        self.predict_mode = predict_mode

    def __call__(self, features: List[Tuple]) -> Dict[str, Any]:
        tmp_ds = LogDataset(
            features,
            vocab=self.vocab,
            seq_len=self.seq_len,
            predict_mode=self.predict_mode,
            mask_ratio=self.mask_ratio,
            use_mlm=self.use_mlm,
        )
        # Transform raw tuples into dataset items (k, k_label, d, y)
        items = [tmp_ds[i] for i in range(len(features))]
        batch = tmp_ds.collate_fn(items)
        return {
            "input_ids": batch["bert_input"].to(torch.long),
            "mlm_labels": batch["bert_label"].to(torch.long),
            "device_ids": batch["device_input"].to(torch.long),
            "labels": batch["seq_label"].to(torch.long),
            "window_5min_end": batch["window_5min_end"].to(torch.long),
        }
