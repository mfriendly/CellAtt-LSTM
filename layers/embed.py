"""Embedding layers shared across models.

- PositionalEmbedding: sinusoidal positional encoding
- PatchEmbedding: for PatchTST
- DataEmbedding_inverted: for iTransformer
"""
import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    """Patch + linear projection + positional embedding for PatchTST."""
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*nvars, seq_len]  (already permuted by caller)
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_inverted(nn.Module):
    """Inverted embedding for iTransformer: treats each variable as a token."""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [B, T, N]  →  [B, N, d_model]
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)
