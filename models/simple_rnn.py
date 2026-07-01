"""Vanilla LSTM and GRU baselines for TriAtt26.

Simple encoder → linear-projection decoder (no attention, no curriculum learning).
    forward(x, target=None, global_step=None) -> [B, pred_len, 1]
"""
import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        self.pred_len = args.future_steps
        self.input_proj = nn.Linear(args.in_channels, H)
        self.encoder = nn.LSTM(H, H, num_layers=2, batch_first=True,
                               dropout=args.dropout)
        self.decoder = nn.LSTM(1, H, num_layers=2, batch_first=True,
                               dropout=args.dropout)
        self.fc = nn.Linear(H, 1)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, x, target=None, global_step=None):
        """x: [B, T, C] → [B, pred_len, 1]"""
        x = self.input_proj(x)
        enc_out, (h, c) = self.encoder(x)
        # autoregressive decode
        dec_input = torch.zeros(x.size(0), 1, 1, device=x.device, dtype=x.dtype)
        outputs = []
        for _ in range(self.pred_len):
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            step_out = self.fc(self.drop(dec_out))
            outputs.append(step_out)
            dec_input = step_out
        return torch.cat(outputs, dim=1)


class SimpleGRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        self.pred_len = args.future_steps
        self.input_proj = nn.Linear(args.in_channels, H)
        self.encoder = nn.GRU(H, H, num_layers=2, batch_first=True,
                              dropout=args.dropout)
        self.decoder = nn.GRU(1, H, num_layers=2, batch_first=True,
                              dropout=args.dropout)
        self.fc = nn.Linear(H, 1)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, x, target=None, global_step=None):
        """x: [B, T, C] → [B, pred_len, 1]"""
        x = self.input_proj(x)
        enc_out, h = self.encoder(x)
        dec_input = torch.zeros(x.size(0), 1, 1, device=x.device, dtype=x.dtype)
        outputs = []
        for _ in range(self.pred_len):
            dec_out, h = self.decoder(dec_input, h)
            step_out = self.fc(self.drop(dec_out))
            outputs.append(step_out)
            dec_input = step_out
        return torch.cat(outputs, dim=1)
