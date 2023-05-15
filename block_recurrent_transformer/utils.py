import torch
import torch.nn as nn
import math
import copy
import numpy as np
from typing import Optional
from einops import rearrange, repeat, pack, unpack

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def relative_positions_np(num_queries: int, num_keys: int,
                          offset: Optional[int] = None):
    """Returns a numpy array of relative positions between query and key.

    If num_keys >= num_queries, e.g. for transformer XL or sliding window,
    then offset should be (num_keys - num_queries) to make the last N queries
    line up with the last N keys.  This is the default if offset is None.

    Args:
      num_queries: Number of queries.
      num_keys:    Number of keys.
      offset:      Offset of the first query wrt. to the first key.

    Returns:
    A /numpy/ array of shape [num_queries, num_keys] with the signed distance
    from each query to each key.
    """

    # Get the offset of each query wrt. to each key.
    # If not specified, assume the last N queries line up with the last N keys.
    if offset is None:
        if num_keys < num_queries:
             raise ValueError("Number of keys %d must be greater than queries %d" %
                             (num_keys, num_queries))
    offset = num_keys - num_queries
    qidx = np.arange(0, num_queries, dtype=np.int32).reshape(num_queries, 1)
    kidx = np.arange(0, num_keys, dtype=np.int32).reshape(1, num_keys)
    return kidx - (qidx + offset)


def sliding_window_mask(num_queries: int, num_keys: int, window_length: int = 0):
    """Returns a causal mask of the same shape as attn."""

    # The mask ranges over the window_length positions prior to          dd each query.
    if window_length == 0:
        window_length = num_queries

    kqpos = relative_positions_np(num_queries, num_keys)  # 2D mask

    # The causal mask includes only those tokens *before* the current token.
    # This slightly improves perplexity in practice, and simplifies generation.
    # Each token attends to exactly window_length prior tokens.
    mask = (kqpos < 0) & (kqpos >= -window_length)
    return mask

def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        scale_base = 512,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)#dim一半长度的递增数组，最大值接近1
        self.register_buffer('scale', scale, persistent = False)

        self.register_buffer('cached_freqs', None, persistent = False)
        self.register_buffer('cached_scales', None, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        device = self.device

        if exists(self.cached_freqs):
            cached_seq_len = self.cached_freqs.shape[-2]
            if cached_seq_len >= seq_len:
                return self.cached_freqs[:seq_len], self.cached_scales[:seq_len]

        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        self.register_buffer('cached_freqs', freqs, persistent = False)
        self.register_buffer('cached_scales', scale, persistent = False)
        # print(freqs.shape, scale.shape)
        return freqs, scale#freqs是存储角度(弧度)的数组:格式为:[0～0,0～0],[1～0,1～0],[2～0,2～0],[3～0,3～0]
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, pos, scale = 1., device = None):#使用旋转位置编码
    # print(device)
    scale = default(scale, 1.)
    scale = scale.to(device)
    seq_len = t.shape[-2]
    # print(pos.shape[-2], seq_len)
    assert pos.shape[-2] >= seq_len

    pos = pos[-seq_len:].to(device)

    if isinstance(scale, torch.Tensor):
        assert scale.shape[-2] >= seq_len
        scale = scale[-seq_len:]
    # print(t.device, rotate_half(t).device, pos.cos().device, pos.sin().device, scale.device)
    return (t * pos.cos() * scale) + (rotate_half(t).to(device) * pos.sin() * scale)#“旋转操作”也称为“复数乘法”，余弦（实部），正弦（虚部）
