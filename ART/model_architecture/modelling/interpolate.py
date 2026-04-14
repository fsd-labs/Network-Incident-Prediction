import torch 
import torch.nn as nn
import torch.nn.functional as F


            
class Interpolate(nn.Module):
    """
    Phiên bản không có learnable weights - chỉ dùng thống kê thuần túy
    Sử dụng uniform filter và repeat/aggregate theo cửa sổ trượt
    """
    def __init__(self, max_len_seq: int):
        super().__init__()
        self.max_len_seq = max_len_seq -1
    
    def normalize_sequence(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        Chuẩn hóa sequence về chiều max_len_seq
        Args:
            x: Input tensor [B, L, H]
            attn_mask: Attention mask [B, L] where 1=valid, 0=padding
        Returns:
            Normalized tensor [B, max_len_seq, H]
        """
        batch_size, seq_length, hidden_dim = x.shape
        
        # Bỏ token <SOS> ở vị trí 0
        x = x[:, 1:, :]  # [B, L-1, H]
        if attn_mask is not None:
            attn_mask = attn_mask[:, 1:]  # [B, L-1]
        
        # Cập nhật seq_length sau khi bỏ CLS
        seq_length = seq_length - 1
        
        if seq_length == self.max_len_seq:
            # Đã đúng chiều rồi
            return x
        
        elif seq_length > self.max_len_seq:
            # Downsample: average pooling từ seq_length về max_len_seq (vectorized)
            if attn_mask is not None:
                # Tính indices cho mỗi window
                deviation = seq_length / self.max_len_seq
                indices = torch.floor(torch.arange(seq_length, device=x.device) / deviation).long()
                indices = torch.clamp(indices, 0, self.max_len_seq - 1)
                
                # Apply mask
                x_masked = x * attn_mask.unsqueeze(-1)  # [B, L, H]
                
                # Cộng gộp theo indices (vectorized với scatter_add)
                # Expand indices: [L] -> [B, L, H]
                indices_expanded = indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_dim)
                
                indices_expanded = indices_expanded.contiguous()
                x_masked = x_masked.contiguous()
                
                output = torch.zeros(batch_size, self.max_len_seq, hidden_dim, device=x.device, dtype=x.dtype)
                output.scatter_add_(1, indices_expanded, x_masked)
                
                # Đếm số lượng mask cho mỗi position
                indices_mask = indices.unsqueeze(0).expand(batch_size, -1).contiguous()  # [B, L]
                mask_count = torch.zeros(batch_size, self.max_len_seq, device=x.device, dtype=attn_mask.dtype)
                mask_count.scatter_add_(1, indices_mask, attn_mask.contiguous())
                
                # Average
                mask_count = mask_count.unsqueeze(-1).clamp_min(1e-6)
                output = output / mask_count
            else:
                # Không có mask: dùng adaptive_avg_pool1d
                # Reshape: [B, L, H] -> [B*H, L]
                x_reshaped = x.permute(0, 2, 1).reshape(batch_size * hidden_dim, seq_length)
                pooled = F.adaptive_avg_pool1d(x_reshaped, self.max_len_seq)
                output = pooled.reshape(batch_size, hidden_dim, self.max_len_seq).permute(0, 2, 1)
            
        else:
            # Upsample: repeat theo // và pad phần dư bằng 0
            repeat = self.max_len_seq // seq_length
            remainder = self.max_len_seq % seq_length

            # Lặp mỗi position 'repeat' lần
            output = x.repeat_interleave(repeat, dim=1)  # [B, seq_length * repeat, H]

            # Pad 0 cho phần dư nếu có
            if remainder > 0:
                pad = torch.zeros(batch_size, remainder, hidden_dim, device=x.device, dtype=x.dtype)
                output = torch.cat([output, pad], dim=1)  # [B, max_len_seq, H]
        
        return output

