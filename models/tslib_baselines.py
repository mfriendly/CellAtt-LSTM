"""TSLib baseline wrappers adapted to TriAtt26 interface.

Each wrapper follows TriAtt26 model contract:
    __init__(args)
    forward(x, target=None, global_step=None) -> [B, pred_len, 1]
    x: [B, seq_len, n_features]

Wraps TSLib models whose native interface is:
    forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, pred_len, enc_in]
"""
import argparse
import torch
import torch.nn as nn

# Must match tslib_modules/layers/Embed.py freq_map exactly
FREQ_DIM_MAP = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 4, 'b': 3}

from tslib_modules.models.DLinear import Model as DLinearModel
from tslib_modules.models.Autoformer import Model as AutoformerModel
from tslib_modules.models.TimesNet import Model as TimesNetModel
from tslib_modules.models.TimeMixer import Model as TimeMixerModel
from tslib_modules.models.TimeXer import Model as TimeXerModel
from tslib_modules.models.TiDE import Model as TiDEModel


# ──────────────────────── Config adapter ────────────────────────
def _make_tslib_config(args, **overrides):
    """Map TriAtt26 flat args → TSLib-compatible config namespace."""
    cfg = argparse.Namespace(
        # core dimensions
        seq_len=args.past_steps,
        label_len=args.past_steps // 2,
        pred_len=args.future_steps,
        enc_in=args.in_channels,
        dec_in=args.in_channels,
        c_out=1,
        # task
        task_name="long_term_forecast",
        output_attention=False,
        use_norm=True,
        # transformer
        d_model=getattr(args, "d_model", 64),
        n_heads=getattr(args, "n_heads", 8),
        e_layers=getattr(args, "e_layers", 2),
        d_layers=getattr(args, "d_layers", 1),
        d_ff=getattr(args, "d_ff", 256),
        dropout=args.dropout,
        activation=getattr(args, "activation", "gelu"),
        factor=getattr(args, "factor", 1),
        # embeddings
        embed="timeF",
        freq={"weekly": "w", "daily": "d", "hourly": "h", "monthly": "m"}.get(
            getattr(args, "granularity", "weekly"), "w"),
        # decomposition / moving average
        moving_avg=getattr(args, "moving_avg", 25),
        # PatchTST / Crossformer
        patch_len=getattr(args, "patch_len", 4),
        stride=getattr(args, "patch_stride", 2),
        seg_len=getattr(args, "seg_len", 6),
        win_size=getattr(args, "win_size", 2),
        # TimesNet
        top_k=getattr(args, "top_k", 5),
        num_kernels=getattr(args, "num_kernels", 6),
        # TimeMixer
        down_sampling_window=getattr(args, "down_sampling_window", 2),
        down_sampling_layers=getattr(args, "down_sampling_layers", 3),
        down_sampling_method=getattr(args, "down_sampling_method", "avg"),
        channel_independence=getattr(args, "channel_independence", 0),
        decomp_method=getattr(args, "decomp_method", "moving_avg"),
        # TFT
        hidden_size=getattr(args, "hidden_size", 64),
        lstm_layers=getattr(args, "lstm_layers", 1),
        attention_head_size=getattr(args, "attention_head_size", 4),
        data="custom",
        # CrossLinear
        alpha=getattr(args, "alpha", 1.0),
        beta=getattr(args, "beta", 0.5),
        features=getattr(args, "features", "MS"),
        class_strategy="projection",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ──────────────────────── Base wrapper ────────────────────────
class _TSLibBase(nn.Module):
    """Base wrapper: handles dummy marks, forward dispatch, channel select.

    TSLib models are always kept in float32 to avoid internal dtype mismatches.
    Input is cast to float32, output is cast back to the caller's dtype.
    """

    def __init__(self, args, tslib_cls, **config_overrides):
        super().__init__()
        self.pred_len = args.future_steps
        cfg = _make_tslib_config(args, **config_overrides)
        self.label_len = cfg.label_len
        self.tslib_model = tslib_cls(cfg).float()
        self.mark_dim = FREQ_DIM_MAP.get(cfg.freq, 2)

    def _apply(self, fn):
        """Override _apply so tslib_model stays float32 after any dtype cast."""
        super()._apply(fn)
        self.tslib_model = self.tslib_model.float()
        return self

    def forward(self, x, target=None, global_step=None):
        """x: [B, T, C] → [B, pred_len, 1]."""
        B, T, C = x.shape
        device = x.device
        orig_dtype = x.dtype
        # TSLib internals assume float32
        x = x.float()
        md = self.mark_dim
        x_mark_enc = torch.zeros(B, T, md, device=device)
        # Decoder input: last label_len encoder tokens + zeros for pred horizon
        label_ctx = x[:, -self.label_len:, :]
        dec_zeros = torch.zeros(B, self.pred_len, C, device=device)
        x_dec = torch.cat([label_ctx, dec_zeros], dim=1)
        x_mark_dec = torch.zeros(B, self.label_len + self.pred_len, md, device=device)
        out = self.tslib_model(x, x_mark_enc, x_dec, x_mark_dec)
        # out: [B, pred_len, C] → take target channel (idx 0), restore dtype
        return out[:, :, 0:1].to(orig_dtype)


# ──────────────────────── Concrete wrappers ────────────────────────
class DLinear(_TSLibBase):
    def __init__(self, args):
        super().__init__(args, DLinearModel)


class Autoformer(_TSLibBase):
    def __init__(self, args):
        super().__init__(args, AutoformerModel)


class TimesNet(_TSLibBase):
    def __init__(self, args):
        super().__init__(args, TimesNetModel)


class TimeMixer(_TSLibBase):
    """TimeMixer: pass x_mark=None to avoid pool-vs-slice length mismatch
    when seq_len is not perfectly divisible by 2^down_sampling_layers."""
    def __init__(self, args):
        super().__init__(args, TimeMixerModel)

    def forward(self, x, target=None, global_step=None):
        B, T, C = x.shape
        device = x.device
        orig_dtype = x.dtype
        x = x.float()
        label_ctx = x[:, -self.label_len:, :]
        dec_zeros = torch.zeros(B, self.pred_len, C, device=device)
        x_dec = torch.cat([label_ctx, dec_zeros], dim=1)
        out = self.tslib_model(x, None, x_dec, None)
        return out[:, :, 0:1].to(orig_dtype)


class TimeXer(_TSLibBase):
    def __init__(self, args):
        super().__init__(args, TimeXerModel)

    def forward(self, x, target=None, global_step=None):
        """TimeXer may output [B, pred_len, 1] directly for c_out=1."""
        out = super().forward(x, target, global_step)
        if out.dim() == 2:
            out = out.unsqueeze(-1)
        return out[:, :, 0:1]


class TiDE(_TSLibBase):
    def __init__(self, args):
        super().__init__(args, TiDEModel)
