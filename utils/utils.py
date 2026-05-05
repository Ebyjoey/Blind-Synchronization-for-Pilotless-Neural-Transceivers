"""
utils.py — Seeds, device selection, logging helpers.
"""

import os
import random
import numpy as np
import torch


# ── Global device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snr_to_noise_std(snr_db: float, rate: float = 1.0) -> float:
    """
    Convert SNR (dB) to noise standard deviation.
    Assumes unit-average-power signal.
    rate = code rate (bits/symbol), used to normalise Eb/N0 vs Es/N0.
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / (2.0 * snr_lin * rate)
    return float(np.sqrt(noise_var))


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """Bit-error rate between two equal-length bit arrays."""
    n = min(len(bits_tx), len(bits_rx))
    return float(np.mean(bits_tx[:n] != bits_rx[:n]))


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)
