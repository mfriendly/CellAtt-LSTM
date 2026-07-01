"""CARD (Channel Aligned Robust Blend Transformer) wrapper.

Source: https://github.com/wxie9/CARD
Adapted to TriAtt26 interface:
    __init__(args)
    forward(x, target=None, global_step=None) -> [B, pred_len, 1]
    x: [B, seq_len, n_features]
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class CARDAttention(nn.Module):
    def __init__(self, config, over_hidden=False):
        super().__init__()
        self.over_hidden = over_hidden
        self.n_heads = config.n_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.n_heads
        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear(config.d_model, config.d_model)
        self.norm_post1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model, momentum=config.momentum), Transpose(1, 2))
        self.norm_post2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model, momentum=config.momentum), Transpose(1, 2))
        self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model, momentum=config.momentum), Transpose(1, 2))
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        self.ff_1 = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.d_ff, config.d_model, bias=True))
        self.ff_2 = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.d_ff, config.d_model, bias=True))
        self.merge_size = config.merge_size
        ema_size = max(config.enc_in, config.total_token_number, config.dp_rank)
        ema_matrix = torch.zeros((ema_size, ema_size))
        alpha = config.alpha
        ema_matrix[0][0] = 1
        for i in range(1, config.total_token_number):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i - 1][j] * (1 - alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer('ema_matrix', ema_matrix)

    def ema(self, src):
        return torch.einsum('bnhad,ga->bnhgd', src, self.ema_matrix[:src.shape[-2], :src.shape[-2]])

    def dynamic_projection(self, src, mlp):
        src_dp = F.softmax(mlp(src), dim=-1)
        return torch.einsum('bnhef,bnhec->bnhcf', src, src_dp)

    def forward(self, src, *args, **kwargs):
        B, nvars, H, C = src.shape
        qkv = self.qkv(src).reshape(B, nvars, H, 3, self.n_heads, C // self.n_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if not self.over_hidden:
            attn_score = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k)) / self.head_dim ** -0.5
            attn = self.attn_dropout(F.softmax(attn_score, dim=-1))
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn, v)
        else:
            v_dp, k_dp = self.dynamic_projection(v, self.dp_v), self.dynamic_projection(k, self.dp_k)
            attn_score = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k_dp)) / self.head_dim ** -0.5
            attn = self.attn_dropout(F.softmax(attn_score, dim=-1))
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn, v_dp)
        attn_score_h = torch.einsum('bnhae,bnhaf->bnhef', q, k) / q.shape[-2] ** -0.5
        attn_h = self.attn_dropout(F.softmax(attn_score_h, dim=-1))
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_h, v)
        ms = self.merge_size
        output1 = rearrange(output_along_token.reshape(B * nvars, -1, self.head_dim), 'bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d', hl1=self.n_heads // ms, hl2=output_along_token.shape[-2], hl3=ms).reshape(B * nvars, -1, self.head_dim * self.n_heads)
        output2 = rearrange(output_along_hidden.reshape(B * nvars, -1, self.head_dim), 'bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d', hl1=self.n_heads // ms, hl2=output_along_token.shape[-2], hl3=ms).reshape(B * nvars, -1, self.head_dim * self.n_heads)
        output1 = self.norm_post1(output1).reshape(B, nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2).reshape(B, nvars, -1, self.n_heads * self.head_dim)
        src2 = self.ff_1(output1) + self.ff_2(output2)
        src = src + src2
        src = self.norm_attn(src.reshape(B * nvars, -1, self.n_heads * self.head_dim)).reshape(B, nvars, -1, self.n_heads * self.head_dim)
        return src


class CARDformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.task_name = config.task_name
        patch_num = int((config.seq_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num, config.d_model) * 1e-2)
        self.model_token_number = 0
        self.total_token_number = patch_num + self.model_token_number + 1
        config.total_token_number = self.total_token_number
        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.use_statistic = config.use_statistic
        self.W_statistic = nn.Linear(2, config.d_model)
        self.cls = nn.Parameter(torch.randn(1, config.d_model) * 1e-2)
        self.W_out = nn.Linear((patch_num + 1 + self.model_token_number) * config.d_model, config.pred_len)
        self.Attentions_over_token = nn.ModuleList([CARDAttention(config) for _ in range(config.e_layers)])
        self.Attentions_over_channel = nn.ModuleList([CARDAttention(config, over_hidden=True) for _ in range(config.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(config.d_model, config.d_model) for _ in range(config.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(config.dropout) for _ in range(config.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model, momentum=config.momentum), Transpose(1, 2)) for _ in range(config.e_layers)])

    def forward(self, z):
        b, c, s = z.shape
        z_mean = torch.mean(z, dim=-1, keepdim=True)
        z_std = torch.std(z, dim=-1, keepdim=True)
        z = (z - z_mean) / (z_std + 1e-4)
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embed
        if self.use_statistic:
            z_stat = torch.cat((z_mean, z_std), dim=-1)
            if z_stat.shape[-2] > 1:
                z_stat = (z_stat - torch.mean(z_stat, dim=-2, keepdim=True)) / (torch.std(z_stat, dim=-2, keepdim=True) + 1e-4)
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2), z_embed), dim=-2)
        else:
            cls_token = self.cls.repeat(z_embed.shape[0], z_embed.shape[1], 1, 1)
            z_embed = torch.cat((cls_token, z_embed), dim=-2)
        inputs = z_embed
        b2, c2, t, h = inputs.shape
        for a_2, a_1, mlp, drop, norm in zip(self.Attentions_over_token, self.Attentions_over_channel, self.Attentions_mlp, self.Attentions_dropout, self.Attentions_norm):
            output_1 = a_1(inputs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            output_2 = a_2(output_1)
            outputs = drop(mlp(output_1 + output_2)) + inputs
            outputs = norm(outputs.reshape(b2 * c2, t, -1)).reshape(b2, c2, t, -1)
            inputs = outputs
        z_out = self.W_out(outputs.reshape(b, c, -1))
        z = z_out * (z_std + 1e-4) + z_mean
        return z


class CARD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.future_steps
        cfg = argparse.Namespace(
            seq_len=args.past_steps,
            pred_len=args.future_steps,
            enc_in=args.in_channels,
            d_model=getattr(args, "card_d_model", 16),
            d_ff=getattr(args, "card_d_ff", 32),
            n_heads=getattr(args, "card_n_heads", 2),
            e_layers=getattr(args, "card_e_layers", 2),
            dropout=getattr(args, "card_dropout", 0.3),
            patch_len=getattr(args, "card_patch_len", 16),
            stride=getattr(args, "card_stride", 8),
            dp_rank=getattr(args, "card_dp_rank", 8),
            merge_size=getattr(args, "card_merge_size", 2),
            momentum=getattr(args, "card_momentum", 0.1),
            alpha=getattr(args, "card_alpha", 0.9),
            use_statistic=getattr(args, "card_use_statistic", False),
            task_name="long_term_forecast",
            total_token_number=0,
        )
        self.model = CARDformer(cfg).float()

    def _apply(self, fn):
        super()._apply(fn)
        self.model = self.model.float()
        return self

    def forward(self, x, target=None, global_step=None):
        """x: [B, T, C] → [B, pred_len, 1]."""
        orig_dtype = x.dtype
        x = x.float()
        x = x.permute(0, 2, 1)
        out = self.model(x)
        out = out.permute(0, 2, 1)
        return out[:, :, 0:1].to(orig_dtype)
