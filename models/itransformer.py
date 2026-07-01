"""iTransformer — Inverted Transformer for time series forecasting.

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective
for Time Series Forecasting" (ICLR 2024)
https://arxiv.org/abs/2310.06625

Core idea: each *variable* (feature) is treated as a token, and attention
is applied across variables instead of across time steps.
"""
import torch
import torch.nn as nn

from layers.attention import FullAttention, AttentionLayer
from layers.embed import DataEmbedding_inverted
from layers.transformer import TransformerEncoderLayer, TransformerEncoder


class iTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.past_steps
        self.pred_len = args.future_steps
        self.in_channels = args.in_channels

        self.enc_embedding = DataEmbedding_inverted(
            args.past_steps, args.d_model, dropout=args.dropout)

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

        self.projection = nn.Linear(args.d_model, args.future_steps)

    def forward(self, x, target=None, global_step=None):
        """x: [B, T, C]. Returns [B, pred_len, 1]."""
        # Instance normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # Inverted embedding: [B, T, C] -> [B, C, d_model]
        enc_out = self.enc_embedding(x)

        # Attention across variables
        enc_out, _ = self.encoder(enc_out)

        # Project each variable to pred_len: [B, C, pred_len]
        dec_out = self.projection(enc_out)
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, C]

        # De-normalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out[:, :, 0:1]  # [B, pred_len, 1]
