from collections import defaultdict
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class LogDataset(Dataset):
    def __init__(self, log_corpus, vocab, seq_len,
                 corpus_lines=None, encoding="utf-8",
                 on_memory=True, predict_mode=False,
                 mask_ratio=0.15, use_mlm=False):
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.encoding = encoding
        self.predict_mode = predict_mode
        self.log_corpus = log_corpus
        self.corpus_lines = len(log_corpus)
        self.use_mlm = use_mlm
        self.mask_ratio = (mask_ratio if use_mlm else 0.0)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        item = self.log_corpus[idx]

        # normalize tuple:
        if len(item) == 4:
            k, _t_ignored, d, y = item
        elif len(item) == 3:
            k, _t_ignored, d = item
            y = -1
        elif len(item) == 2:
            k, d = item
            y = -1
        else:
            raise ValueError(f"Unexpected item format at idx {idx}: len={len(item)}")

        k_masked, k_label = self.random_item(k)

        # prepend SOS
        k = [self.vocab.sos_index] + k_masked
        k_label = [self.vocab.pad_index] + k_label

        # device align
        dev_pad = 0
        dev_cls = d[0] if len(d) > 0 else dev_pad
        d = [dev_cls] + d[:len(k_masked)]
        if len(d) < len(k):
            d += [self.vocab.pad_index] * (len(k) - len(d))

        _t_ignored = _t_ignored[0] if _t_ignored else []

        return k, k_label, d, int(y), _t_ignored

    def _to_id(self, tok):
        if isinstance(tok, int):
            return tok if 0 <= tok < len(self.vocab) else self.vocab.unk_index
        return self.vocab.stoi.get(tok, self.vocab.unk_index)

    def random_item(self, k):
        if (not self.use_mlm) or self.mask_ratio <= 0:
            tokens = [self._to_id(tok) for tok in k]
            return tokens, [0] * len(k)

        tokens = list(k)
        output_label = []
        for i, tok in enumerate(tokens):
            prob = random.random()
            if prob < self.mask_ratio:
                if self.predict_mode:
                    tokens[i] = self.vocab.mask_index
                    output_label.append(self._to_id(tok))
                    continue
                prob /= self.mask_ratio
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                else:
                    tokens[i] = self._to_id(tok)
                output_label.append(self._to_id(tok))
            else:
                tokens[i] = self._to_id(tok)
                output_label.append(0)
        return tokens, output_label

    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        lens = [len(seq[0]) for seq in batch]
        if dynamical_pad:
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            seq_len = self.seq_len

        out = defaultdict(list)
        for (k, k_label, d, y, _t_ignored) in batch:
            k = k[:seq_len]
            k_label = k[1:seq_len+1] if len(k) > 1 else [self.vocab.pad_index]
            if len(k_label) < len(k):
                k_label.append(self.vocab.eos_index)
            d = d[:seq_len]

            # padding
            pad = [self.vocab.pad_index] * (seq_len - len(k))
            k += pad
            k_label += pad
            d += [self.vocab.pad_index] * (seq_len - len(d))
            out["bert_input"].append(k)
            out["bert_label"].append(k_label)
            out["device_input"].append(d)
            out["seq_label"].append(int(y))
            out["window_5min_end"].append(_t_ignored)

        return {
            "bert_input": torch.tensor(out["bert_input"], dtype=torch.long),
            "bert_label": torch.tensor(out["bert_label"], dtype=torch.long),
            "device_input": torch.tensor(out["device_input"], dtype=torch.long),
            "seq_label": torch.tensor(out["seq_label"], dtype=torch.long),
            "window_5min_end": torch.tensor(out["window_5min_end"], dtype=torch.long),
        }
