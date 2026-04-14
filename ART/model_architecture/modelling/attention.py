import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None, causal: bool = False):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_mask = ~mask
            else:
                attn_mask = torch.isneginf(mask)
        
            # Normalize common shapes (broadcastable to [B, h, L, S])
            if attn_mask.dim() == 2:           # [L, S]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,L,S]
            elif attn_mask.dim() == 3:         # [B, L, S]
                attn_mask = attn_mask.unsqueeze(1)               # [B,1,L,S]
        
        out = F.scaled_dot_product_attention(
            query, key, value,
            # attn_mask=attn_mask,
            dropout_p=(dropout.p if (dropout is not None and self.training) else 0.0),
            is_causal=causal
        )

        return out, None
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, causal: bool = False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout, causal = causal)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
