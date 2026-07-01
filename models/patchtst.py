"""PatchTST — Patch-based Time Series Transformer.

Reference: Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023)
https://arxiv.org/abs/2211.14730
"""
import torch
import torch.nn as nn

from layers.attention import FullAttention, AttentionLayer
from layers.embed import PatchEmbedding
from layers.transformer import TransformerEncoderLayer, TransformerEncoder


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [B, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class PatchTST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.past_steps
        self.pred_len = args.future_steps
        patch_len = args.patch_len
        stride = args.patch_stride
        padding = stride

        self.patch_embedding = PatchEmbedding(
            args.d_model, patch_len, stride, padding, args.dropout)

        self.encoder = TransformerEncoder(
            [TransformerEncoderLayer(
                AttentionLayer(
                    FullAttention(False, args.factor,
                                 attention_dropout=args.dropout,
                                 output_attention=False),
                    args.d_model, args.n_heads),
                args.d_model, args.d_ff, dropout=args.dropout,
                activation=args.activation)
             for _ in range(args.e_layers)],
            norm_layer=nn.LayerNorm(args.d_model))

        self.head_nf = args.d_model * int(
            (args.past_steps - patch_len) / stride + 2)
        self.head = FlattenHead(
            args.in_channels, self.head_nf, args.future_steps,
            head_dropout=args.dropout)

    def forward(self, x, target=None, global_step=None):
        """x: [B, T, C]. Returns [B, pred_len, 1]."""
        # Instance normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # Patch embedding
        x = x.permute(0, 2, 1)  # [B, C, T]
        enc_out, n_vars = self.patch_embedding(x)

        # Encoder
        enc_out, _ = self.encoder(enc_out)

        # Reshape and predict
        enc_out = enc_out.reshape(
            -1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, C, d_model, patch_num]
        dec_out = self.head(enc_out)  # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, C]

        # De-normalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out[:, :, 0:1]  # [B, pred_len, 1]
