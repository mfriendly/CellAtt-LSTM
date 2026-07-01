"""TFT — Simplified Temporal Fusion Transformer.

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (IJoF 2021)
https://arxiv.org/abs/1912.09363

Simplified version: no static covariates, no known future inputs.
Components: GRN → Variable Selection → LSTM enc/dec →
            Interpretable Multi-Head Attention → Gated output.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_output)
        self.glu = GatedLinearUnit(d_output)
        self.norm = nn.LayerNorm(d_output)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_input, d_output) if d_input != d_output else None

    def forward(self, x):
        residual = self.skip(x) if self.skip else x
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_vars, d_hidden, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_hidden, d_hidden, dropout)
            for _ in range(n_vars)])
        self.weight_grn = GatedResidualNetwork(
            n_vars * d_hidden, d_hidden, n_vars, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, T, n_vars]
        var_outputs = []
        for i in range(self.n_vars):
            var_outputs.append(self.var_grns[i](x[:, :, i:i+1]))
        var_stack = torch.stack(var_outputs, dim=-1)  # [B, T, H, n_vars]
        flat = torch.cat(var_outputs, dim=-1)  # [B, T, n_vars*H]
        weights = self.softmax(self.weight_grn(flat))  # [B, T, n_vars]
        selected = (var_stack * weights.unsqueeze(2)).sum(dim=-1)  # [B, T, H]
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, L, _ = q.shape
        S = k.shape[1]
        H = self.n_heads
        q = self.W_q(q).view(B, L, H, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, S, H, self.d_k).transpose(1, 2)
        v = self.W_v(v)  # shared across heads for interpretability [B, S, D]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn_avg = attn.mean(dim=1)  # [B, L, S]
        out = torch.matmul(attn_avg, v)  # [B, L, D]
        return self.W_o(out)


class TFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        C = args.in_channels
        self.pred_len = args.future_steps

        self.vsn = VariableSelectionNetwork(C, H, args.dropout)
        self.encoder_lstm = nn.LSTM(H, H, batch_first=True, num_layers=1)
        self.decoder_lstm = nn.LSTM(1, H, batch_first=True, num_layers=1)
        self.attention = InterpretableMultiHeadAttention(H, args.n_heads, args.dropout)
        self.gate = GatedLinearUnit(H)
        self.norm = nn.LayerNorm(H)
        self.output_proj = nn.Linear(H, 1)

    def forward(self, x, target=None, global_step=None):
        """x: [B, T, C]. Returns [B, pred_len, 1]."""
        B = x.shape[0]
        selected, _ = self.vsn(x)  # [B, T, H]
        enc_out, (h, c) = self.encoder_lstm(selected)  # [B, T, H]
        dec_input = torch.zeros(B, self.pred_len, 1,
                                device=x.device, dtype=x.dtype)
        dec_out, _ = self.decoder_lstm(dec_input, (h, c))  # [B, pred_len, H]
        attn_out = self.attention(dec_out, enc_out, enc_out)
        out = self.norm(self.gate(attn_out) + dec_out)
        return self.output_proj(out)  # [B, pred_len, 1]
