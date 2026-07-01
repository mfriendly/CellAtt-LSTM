"""
iTransformer model implementation.
Extracted from Time-Series-Library with fixed import paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 1)
        self.output_attention = getattr(configs, 'output_attention', False)
        self.use_norm = getattr(configs, 'use_norm', True)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model,
            getattr(configs, 'embed', 'fixed'),
            getattr(configs, 'freq', 'h'),
            getattr(configs, 'dropout', 0.1)
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, getattr(configs, 'factor', 1),
                                    attention_dropout=getattr(configs, 'dropout', 0.1),
                                    output_attention=self.output_attention),
                        configs.d_model, getattr(configs, 'n_heads', 8)
                    ),
                    configs.d_model,
                    getattr(configs, 'd_ff', None),
                    dropout=getattr(configs, 'dropout', 0.1),
                    activation=getattr(configs, 'activation', 'gelu')
                ) for l in range(getattr(configs, 'e_layers', 2))
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, self.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(getattr(configs, 'dropout', 0.1))
            self.projection = nn.Linear(
                configs.d_model * configs.enc_in, getattr(configs, 'num_class', 2)
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc /= stdev

        _, _, N = x_enc.shape
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # B N E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N L -> B L N
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return dec_out

    def anomaly_detection(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        # B N E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N L -> B L N
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        # B N E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # B N E -> B N * E -> B 1 * (N * E) -> B (N * E)
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
