import torch
import torch.nn as nn
import os
import sys
import math
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))
from modelling.transformer import TransformerBlock
from modelling.embedding import BERTEmbedding
from modelling.interpolate import Interpolate


#=================================BASE BERT MODEL===========================================
class BERT(nn.Module):
    def __init__(self, vocab_size, max_len=20480, hidden=256, n_layers=4, attn_heads=8, dropout=0.1, is_logkey=True, is_time=False, is_device=True, num_devices=None, causal=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 2

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_len=max_len, is_logkey=is_logkey, is_time=is_time, is_device=is_device, num_devices=num_devices)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout, causal) for _ in range(n_layers)])

    def forward(self, x, segment_info=None, time_info=None, device_info=None):
        key_padding = (x > 0)                          # [B, L]
        mask = key_padding[:, None, None, :]           # [B, 1, 1, L]
        x = self.embedding(x, segment_info, time_info, device_info)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x
    

#=================================MODIFIED LOGBERT===========================================
class LDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):  # x: [B, H, L]
        if not self.training or self.p == 0:
            return x
        B, H, L = x.shape
        # mask: [B, 1, L]
        mask = (torch.rand(B, 1, L, device=x.device) > self.p).float()
        return x * mask

class MaskedLogModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class ClassifierHead(nn.Module):
    def __init__(self, hidden, max_len_seq, num_classes=2):
        super().__init__()
        self.hidden = hidden 
        self.max_len_seq = max_len_seq -1
        self.dropout1 = LDropout(0.1)
        self.linear1 = nn.Linear(self.max_len_seq, 1)
        nn.init.constant_(self.linear1.weight, 1.0 / self.max_len_seq)
        self.dropout2 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = x.squeeze(-1)
        return self.linear2(self.dropout2(x))

class BERTLog(nn.Module):
    def __init__(self, bert: BERT, num_classes=2, vocab_size=None, use_mlm=True, max_len_seq=1024):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        :param max_len_seq: chuẩn hóa sequence về chiều này
        """
        super().__init__()
        self.bert = bert
        self.use_mlm = use_mlm
        self.max_len_seq = max_len_seq
        
        if use_mlm:
            assert vocab_size is not None
            self.mask_lm = MaskedLogModel(self.bert.hidden, vocab_size)
        
        # Sử dụng Interpolate2 (không học, chỉ thống kê)
        self.interpolate = Interpolate(max_len_seq=max_len_seq)
        
        # Sử dụng ClassifierHead2 với hidden dim = median_seq_length
        self.classifier = ClassifierHead(self.bert.hidden, max_len_seq=max_len_seq, num_classes=num_classes)
        
        self.result = {"cls_logits": None, "cls_input": None, "logkey_output": None}

    def forward(self, x, segment_label=None, time_input=None, device_input=None):
        input_ids = x
        x = self.bert(x, segment_label, time_input, device_input)  # [B, L, H]

        attn_mask = (input_ids > 0).float()
        attn_mask[:, 0] = 0
        
        # Chuẩn hóa về [B, max seq len, H] bằng Interpolate2
        x_normalized = self.interpolate.normalize_sequence(x, attn_mask=attn_mask)  # [B, 12287, H]
        
        self.result["cls_input"] = x_normalized
        self.result["cls_logits"] = self.classifier(x_normalized)

        if self.use_mlm:
            logkey_output = self.mask_lm(x)
            self.result["logkey_output"] = logkey_output
        
        return self.result
