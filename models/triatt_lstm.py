"""TriAttLSTM — Tri-Attention LSTM with encoder–decoder architecture.

Three attention insertion points controlled by layer_string (4 digits):
  [0] nlayer_geatt1: seq2seq cross-attention in Decoder
  [1] nlayer_geatt2: self-attention on input features
  [2] nlayer_geatt3a: attention inside LSTM cell (combined gate)
  [3] nlayer_geatt3b: attention inside LSTM cell (hidden state)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.attention import GEAttention
from layers.lstm_cell import CustomLSTMCell


def _init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="selu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ───────────────────────── Encoder ─────────────────────────
class Encoder(nn.Module):
    def __init__(self, hidden_size, dropout, dropout_type, activation,
                 nlayer_geatt3a, nlayer_geatt3b, n_heads, add_skips, device,
                 dtype, slim_cell=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.rnn = CustomLSTMCell(
            in_channels=hidden_size, hidden_size=hidden_size,
            dropout=dropout, dropout_type=dropout_type,
            activation=activation, nlayer_geatt3a=nlayer_geatt3a,
            nlayer_geatt3b=nlayer_geatt3b, n_heads=n_heads,
            add_skips=add_skips, slim_cell=slim_cell)

    def forward(self, x, seq_length):
        B, T, D = x.shape
        h = self._init_state(B)
        c = self._init_state(B)
        outputs = []
        for t in range(seq_length):
            inp = x[:, t:t+1, :]
            _, (h, c) = self.rnn(inp, (h, c))
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # [B, T, hidden]

    def _init_state(self, batch_size):
        std = np.sqrt(2.0 / self.hidden_size)
        return Variable(torch.randn(batch_size, self.hidden_size,
                                    dtype=self.dtype, device=self.device) * std)


# ───────────────────────── Decoder ─────────────────────────
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, dropout_type,
                 activation, nlayer_geatt1, nlayer_geatt3a, nlayer_geatt3b,
                 n_heads, add_skips, use_curriculum_learning, cl_decay_steps,
                 dtype, has_exo_cross=False, slim_cell=False):
        super().__init__()
        self.use_cl = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps
        self.dtype = dtype
        self.has_exo_cross = has_exo_cross

        self.rnn = CustomLSTMCell(
            in_channels=1, hidden_size=hidden_size,
            dropout=dropout, dropout_type=dropout_type,
            activation=activation, nlayer_geatt3a=nlayer_geatt3a,
            nlayer_geatt3b=nlayer_geatt3b, n_heads=n_heads,
            add_skips=add_skips, slim_cell=slim_cell)

        self.cross_attn_layers = nn.ModuleList([
            GEAttention(hidden_size, n_heads, dropout)
            for _ in range(nlayer_geatt1)])
        self.attn_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(nlayer_geatt1)])
        self.attn_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(nlayer_geatt1)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout))
            for _ in range(nlayer_geatt1)])

        if has_exo_cross:
            self.exo_attn = GEAttention(hidden_size, n_heads, dropout)
            self.exo_norm = nn.LayerNorm(hidden_size)
            self.exo_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())

        self.context_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_final = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, encoder_outputs, target=None, pred_len=None,
                global_step=None, exo_context=None):
        B = encoder_outputs.shape[0]
        h = encoder_outputs[:, -1, :]
        c = torch.zeros_like(h, dtype=self.dtype)
        dec_input = torch.zeros(B, 1, 1, device=h.device, dtype=self.dtype)
        outputs = []

        for t in range(pred_len):
            _, (h, c) = self.rnn(dec_input, (h, c))
            q = h.unsqueeze(1)  # [B, 1, H]
            attn_out = q
            for attn, n1, n2, ffn in zip(
                    self.cross_attn_layers, self.attn_norms1,
                    self.attn_norms2, self.ffn_layers):
                residual = attn_out
                attn_out = attn(n1(attn_out), encoder_outputs, encoder_outputs)
                attn_out = residual + self.drop(ffn(attn_out))
            out_h = self.context_linear(torch.cat([q, attn_out], dim=-1))
            if self.has_exo_cross and exo_context is not None:
                exo_att = self.exo_attn(
                    self.exo_norm(out_h), exo_context, exo_context)
                gate = self.exo_gate(torch.cat([out_h, exo_att], dim=-1))
                out_h = out_h + gate * exo_att
            out_h = self.drop(out_h)
            out = self.fc_final(out_h)  # [B, 1, 1]
            outputs.append(out)

            if self.training and self.use_cl and target is not None:
                prob = self.cl_decay_steps / (
                    self.cl_decay_steps + np.exp(
                        (global_step or 0) / self.cl_decay_steps))
                if np.random.uniform() < prob and t < target.size(1) - 1:
                    dec_input = target[:, t:t+1, :]
                else:
                    dec_input = out
            else:
                dec_input = out

        return torch.cat(outputs, dim=1)  # [B, pred_len, 1]


# ───────────────────── TriAttLSTM (main) ─────────────────────
class TriAttLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        ls = str(args.layer_string)
        nlayer_geatt1 = int(ls[0])  # decoder cross-attention
        nlayer_geatt2 = int(ls[1])  # input feature self-attention
        nlayer_geatt3a = int(ls[2])  # LSTM cell gate attention
        nlayer_geatt3b = int(ls[3])  # LSTM cell hidden attention

        H = args.hidden_size
        C = args.in_channels
        dtype = torch.float64 if args.dtype == "double" else torch.float32
        self.past_steps = args.past_steps
        self.nlayer_geatt2 = nlayer_geatt2
        self.dtype = dtype
        self.device = args.device
        self.last_skip = args.last_skip
        self.last_linear = args.last_linear
        self.sep_exo = getattr(args, "sep_exo", False) and C > 1

        if self.sep_exo:
            # --- endo/exo separated paths ---
            self.endo_proj = nn.Linear(1, H)
            self.exo_proj = nn.Linear(C - 1, H)
            if nlayer_geatt2 > 0:
                self.feat_attn = nn.ModuleList([
                    GEAttention(H, args.n_heads, args.dropout)
                    for _ in range(nlayer_geatt2)])
                self.feat_norms = nn.ModuleList([
                    nn.LayerNorm(H) for _ in range(nlayer_geatt2)])
            self.exo_norm_out = nn.LayerNorm(H)
        else:
            # --- original joint path ---
            self.input_proj = nn.Linear(C, H)
            if nlayer_geatt2 > 0:
                self.input_proj_res = nn.Linear(C, H)
                self.feat_attn = nn.ModuleList([
                    GEAttention(H, args.n_heads, args.dropout)
                    for _ in range(nlayer_geatt2)])
                self.feat_norms = nn.ModuleList([
                    nn.LayerNorm(H) for _ in range(nlayer_geatt2)])
            self.layer_norm_in = nn.LayerNorm(H)
            self.encoder_proj = nn.Linear(1, H)

        self.drop = nn.Dropout(args.dropout)

        # --- encoder ---
        self.use_native_enc = getattr(args, "native_enc", False)
        if self.use_native_enc:
            n_enc = getattr(args, "pre_enc_layers", 2)
            self.encoder = nn.LSTM(
                H, H, num_layers=n_enc, batch_first=True,
                dropout=args.dropout if n_enc > 1 else 0)
            self.encoder_proj = nn.Linear(1, H)
        else:
            pre_enc = getattr(args, "pre_enc_layers", 0)
            self.pre_enc_layers = pre_enc
            if pre_enc > 0:
                self.pre_encoder = nn.LSTM(
                    H, H, num_layers=pre_enc, batch_first=True,
                    dropout=args.dropout if pre_enc > 1 else 0)
            slim = getattr(args, "slim_cell", False)
            self.custom_encoder = Encoder(
                H, args.dropout, args.dropout_type, args.activations[0],
                nlayer_geatt3a, nlayer_geatt3b, args.n_heads, args.add_skips,
                args.device, dtype, slim_cell=slim)

        # --- decoder ---
        slim = getattr(args, "slim_cell", False)
        self.decoder = Decoder(
            H, args.output_size, args.dropout, args.dropout_type,
            args.activations[0], nlayer_geatt1, nlayer_geatt3a, nlayer_geatt3b,
            args.n_heads, args.add_skips, args.use_curriculum_learning,
            args.cl_decay_steps, dtype, has_exo_cross=self.sep_exo,
            slim_cell=slim)

        # --- skip connection ---
        if self.last_skip and self.last_linear:
            self.linear_out = nn.Linear(args.past_steps, args.future_steps)
            self.skip_logit = nn.Parameter(torch.tensor(-2.2))

        self.apply(_init_kaiming)

    def forward(self, x, target=None, global_step=None):
        """x: [B, past_steps, in_channels], target: [B, future_steps, 1]."""
        original = x

        if self.sep_exo:
            endo = x[:, :, 0:1]
            exo = x[:, :, 1:]
            endo_h = self.endo_proj(endo)
            exo_h = self.exo_proj(exo)
            if self.nlayer_geatt2 > 0:
                for attn, norm in zip(self.feat_attn, self.feat_norms):
                    exo_h = attn(exo_h, exo_h, exo_h) + exo_h
                    exo_h = self.drop(F.gelu(exo_h))
                    exo_h = norm(exo_h)
            exo_ctx = self.exo_norm_out(exo_h)
            enc_in = endo_h
        else:
            exo_ctx = None
            residual = x
            if self.nlayer_geatt2 > 0:
                x = self.input_proj(x)
                for i, (attn, norm) in enumerate(zip(self.feat_attn, self.feat_norms)):
                    x = attn(x, x, x) + self.input_proj_res(residual)
                    x = self.drop(F.gelu(x))
                    x = norm(x)
                    if i == 0 and self.last_skip:
                        x = F.gelu(self.layer_norm_in(x) + original[:, :, 0:1])
            else:
                x = self.input_proj(x)
            enc_in = x

        if self.use_native_enc:
            enc_out, _ = self.encoder(enc_in)
            enc_out = self.drop(self.encoder_proj(enc_out[:, :, 0:1]))
        else:
            if self.pre_enc_layers > 0:
                enc_in, _ = self.pre_encoder(enc_in)
            enc_out = self.custom_encoder(enc_in, self.past_steps)
            if not self.sep_exo:
                enc_out = self.drop(self.encoder_proj(enc_out[:, :, 0:1]))

        pred_len = target.size(1) if target is not None else self.past_steps
        outputs = self.decoder(enc_out, target, pred_len, global_step,
                               exo_context=exo_ctx)

        if self.last_skip:
            if self.last_linear:
                skip = self.linear_out(
                    original[:, :, 0:1].permute(0, 2, 1)).permute(0, 2, 1)
            else:
                skip = original[:, :, 0:1]
            alpha = torch.sigmoid(self.skip_logit)
            outputs = (1 - alpha) * outputs + alpha * skip

        return outputs
