import math
import torch
import torch.nn as nn


def _safe_clamp_indices(indices: torch.Tensor, vocab_size: int, name: str = "input") -> torch.Tensor:
    """Clamp indices to valid range to prevent CUDA illegal memory access.
    
    This is critical for multi-GPU training where out-of-bounds indices
    can cause cudaErrorIllegalAddress errors.
    """
    if indices.numel() == 0:
        return indices
    # Check and clamp to prevent illegal memory access
    max_val = indices.max().item()
    min_val = indices.min().item()
    if max_val >= vocab_size or min_val < 0:
        # Clamp to valid range (0 to vocab_size-1), keeping padding_idx=0 valid
        indices = indices.clamp(min=0, max=vocab_size - 1)
        print(f"Warning: Clamped {name} indices from [{min_val}, {max_val}] to valid range [0, {vocab_size - 1}] to prevent illegal memory access.")
    return indices


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.time_embed = nn.Linear(1, embed_size)

    def forward(self, time_interval):
        return self.time_embed(time_interval)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class DeviceEmbedding(nn.Embedding):
    def __init__(self, num_devices, embed_size=512):
        super().__init__(num_devices, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BertEmbedding
    """
    def __init__(self, vocab_size, embed_size, max_len = 256, dropout=0.1, is_logkey=True, is_time=False, is_device=True, num_devices=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_devices = num_devices
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len = max_len)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        # self.time_embed = TimeEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.is_logkey = is_logkey
        self.is_time = is_time
        # add device embd
        self.is_device = is_device
        if self.is_device:
            assert num_devices is not None
            self.device = DeviceEmbedding(num_devices, embed_size=self.token.embedding_dim)
        else:
            print("NO DEVICE EMBEDDING IN USE")

    def forward(self, sequence, segment_label=None, time_info=None, device_info=None):
        sequence = _safe_clamp_indices(sequence, self.vocab_size, "token")
        
        x = self.token(sequence) + self.position(sequence)
        if segment_label is not None:
            x = x + self.segment(segment_label)
        if self.is_time and time_info is not None:
            x = x + self.time_embed(time_info)
        if self.is_device and device_info is not None:
            device_info = _safe_clamp_indices(device_info, self.num_devices, "device")
            x = x + self.device(device_info)
        return self.dropout(x)
