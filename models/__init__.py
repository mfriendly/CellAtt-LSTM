"""Model registry. All models follow:
    forward(x, target=None, global_step=None) -> [B, pred_len, 1]
    x:      [B, seq_len, n_features]
    target: [B, pred_len, 1]  (only used by TriAttLSTM for curriculum learning)
"""
import copy

from .triatt_lstm import TriAttLSTM
from .patchtst import PatchTST
from .itransformer import iTransformer
from .tft import TFT
from .simple_rnn import SimpleLSTM, SimpleGRU
from .tslib_baselines import (
    DLinear, Autoformer,
    TimesNet, TimeMixer,
    TimeXer, TiDE,
)
from .card import CARD
from .epicolagnn import EpiColaGNN_Wrapper


# ── Ablation factory: thin wrappers that override args for TriAttLSTM ──
def _make_ablation(overrides):
    """Return a TriAttLSTM subclass that patches args before __init__."""
    class _Ablation(TriAttLSTM):
        def __init__(self, args):
            a = copy.copy(args)
            for k, v in overrides.items():
                setattr(a, k, v)
            super().__init__(a)
    return _Ablation


MODEL_REGISTRY = {
    # ── Main ──
    "triatt_lstm": TriAttLSTM,
    "triatt_univar": TriAttLSTM,   # same arch, used with --cfs 0
    # ── Ablations (TriAttLSTM variants) ──
    "triatt_noxattn":    _make_ablation({"layer_string": "0111"}),
    "triatt_nofeattn":   _make_ablation({"layer_string": "1011"}),
    "triatt_nocellattn": _make_ablation({"layer_string": "1100"}),
    "triatt_noskip":     _make_ablation({"last_skip": False}),
    "triatt_nocl":       _make_ablation({"use_curriculum_learning": False}),
    # ── Simple RNN baselines ──
    "lstm": SimpleLSTM,
    "gru": SimpleGRU,
    # ── TriAtt26 native baselines ──
    "patchtst": PatchTST,
    "itransformer": iTransformer,
    "tft": TFT,
    # ── Additional baselines ──
    "card": CARD,
    "epicolagnn": EpiColaGNN_Wrapper,
    # ── TSLib baselines ──
    "dlinear": DLinear,
    "autoformer": Autoformer,
    "timesnet": TimesNet,
    "timemixer": TimeMixer,
    "timexer": TimeXer,
    "tide": TiDE,
}


def build_model(args):
    cls = MODEL_REGISTRY[args.model]
    return cls(args)
