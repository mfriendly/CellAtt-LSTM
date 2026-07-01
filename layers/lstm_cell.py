"""Custom LSTM cell with optional GEAttention and zoneout dropout."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import GEAttention


ACTIVATION_MAP = {
    "tanh": torch.tanh, "relu": F.relu, "selu": F.selu,
    "elu": F.elu, "gelu": F.gelu, "silu": F.silu,
    "softplus": F.softplus, "sigmoid": torch.sigmoid,
    "identity": lambda x: x,
}


def _init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="selu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class CustomLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_size, dropout=0.5,
                 dropout_type="zoneout", activation="gelu",
                 nlayer_geatt3a=0, nlayer_geatt3b=0,
                 n_heads=8, add_skips=False, slim_cell=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.dropout_type = dropout_type
        self.activation_fn = ACTIVATION_MAP[activation]
        self.add_skips = add_skips
        self.nlayer_geatt3a = nlayer_geatt3a
        self.nlayer_geatt3b = nlayer_geatt3b
        self.slim_cell = slim_cell

        drop = nn.Dropout(dropout)

        self.input_block = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.LayerNorm(hidden_size), drop)

        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size), drop)
        self.input_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size), drop)
        self.cell_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size), drop)
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size), drop)

        self.combined_norm = nn.LayerNorm(hidden_size * 2)
        if not slim_cell:
            self.cell_norm = nn.LayerNorm(hidden_size)
            self.hidden_norm = nn.LayerNorm(hidden_size)
        self.input_proj = nn.Linear(in_channels, hidden_size)

        if add_skips:
            self.gate_skip = nn.Linear(hidden_size * 2, hidden_size)

        if nlayer_geatt3a > 0:
            self.geatt_block1 = GEAttention(hidden_size * 2, n_heads, dropout)
        if nlayer_geatt3b > 0:
            self.geatt_block2 = GEAttention(hidden_size, n_heads, dropout)

        self.apply(_init_kaiming)

    def forward(self, x, states):
        """x: [B, 1, in_channels], states: (h, c) each [B, hidden]."""
        h, c = states
        x = x.squeeze(1)  # [B, in_channels]
        B = x.shape[0]
        h = h.view(B, self.hidden_size)
        c = c.view(B, self.hidden_size)

        if x.shape[1] != h.shape[1]:
            x = self.input_proj(x)

        combined = torch.cat([x, h], dim=-1)
        combined = self.activation_fn(self.combined_norm(combined))

        if self.nlayer_geatt3a > 0:
            combined = self.geatt_block1(
                combined.unsqueeze(1), combined.unsqueeze(1),
                combined.unsqueeze(1)).squeeze(1)

        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        g = torch.tanh(self.cell_gate(combined))

        if self.add_skips:
            skip_g = torch.sigmoid(self.gate_skip(combined))
            next_c = (f * c + i * g) * (1 - skip_g) + c * skip_g
        else:
            next_c = f * c + i * g

        if self.slim_cell:
            next_h = o * torch.tanh(next_c)
        else:
            next_c = self.cell_norm(next_c)
            next_h = o * torch.tanh(next_c)
            next_h = self.activation_fn(self.hidden_norm(next_h))

        if self.nlayer_geatt3b > 0:
            next_h = self.geatt_block2(
                next_h.unsqueeze(1), h.unsqueeze(1),
                h.unsqueeze(1)).squeeze(1)

        if self.dropout_type == "zoneout":
            next_h = self._zoneout(h, next_h, self.dropout_rate, self.training)
            next_c = self._zoneout(c, next_c, self.dropout_rate, self.training)

        return next_h, (next_h, next_c)

    @staticmethod
    def _zoneout(prev, nxt, rate, training):
        if training:
            mask = torch.zeros_like(nxt).bernoulli_(rate)
            return mask * prev + (1 - mask) * nxt
        return rate * prev + (1 - rate) * nxt
