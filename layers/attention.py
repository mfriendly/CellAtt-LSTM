"""Attention layers shared across models.

- GEAttention: Graph-Enhanced Attention for TriAttLSTM
- FullAttention: Scaled dot-product attention for PatchTST / iTransformer
- AttentionLayer: Projection wrapper around any inner attention
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GEAttention  (used by TriAttLSTM encoder, decoder, and LSTM cell)
# ---------------------------------------------------------------------------
class GEAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.5):
        super().__init__()
        if d_model % num_heads != 0:
            self.pad = True
            self.original_d_model = d_model
            self.d_model = ((d_model // num_heads) + 1) * num_heads
        else:
            self.pad = False
            self.d_model = d_model
            self.original_d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.original_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        residual = Q
        if self.pad:
            pad_len = self.d_model - self.original_d_model
            Q = F.pad(Q, (0, pad_len))
            K = F.pad(K, (0, pad_len))
            V = F.pad(V, (0, pad_len))
        B, L, _ = Q.shape
        Q = self.W_q(Q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(out)
        return F.gelu(out + residual)


# ---------------------------------------------------------------------------
# FullAttention  (used by PatchTST / iTransformer)
# ---------------------------------------------------------------------------
class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


# ---------------------------------------------------------------------------
# AttentionLayer  (projection wrapper)
# ---------------------------------------------------------------------------
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
