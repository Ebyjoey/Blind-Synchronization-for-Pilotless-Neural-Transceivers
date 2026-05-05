"""
channel.py — All channel models used in this project.

Channels are stateless callables that accept a complex-valued signal tensor
(batch, 2) where dim-1 = [I, Q] and return an impaired version.

Convention
----------
  x  : (B, 2*n)  real-valued  [I0,Q0, I1,Q1, …]
  The encoder outputs n=1 complex symbol represented as 2 reals.
  For multi-symbol blocks, n > 1. All channels preserve shape.
"""

import numpy as np
import torch
import torch.nn as nn
from utils import snr_to_noise_std, DEVICE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """(B, 2n) → (B, n) complex"""
    B = x.shape[0]
    return torch.view_as_complex(x.view(B, -1, 2).contiguous())


def _to_real(x: torch.Tensor) -> torch.Tensor:
    """(B, n) complex → (B, 2n) real"""
    return torch.view_as_real(x).reshape(x.shape[0], -1)


# ── 1. AWGN ──────────────────────────────────────────────────────────────────

class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise.
    snr_db: fixed or sampled per batch if snr_range is given.
    """
    def __init__(self, snr_db: float = 10.0, snr_range=None):
        super().__init__()
        self.snr_db    = snr_db
        self.snr_range = snr_range   # (lo, hi) for random SNR training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.snr_range is not None and self.training:
            lo, hi = self.snr_range
            snr_db = lo + torch.rand(1).item() * (hi - lo)
        else:
            snr_db = self.snr_db

        std = snr_to_noise_std(snr_db)
        noise = torch.randn_like(x) * std
        return x + noise


# ── 2. CFO (Carrier Frequency Offset) ────────────────────────────────────────

class CFOChannel(nn.Module):
    """
    Applies a phase rotation that grows linearly over symbol time:
        y[k] = x[k] * exp(j*2*pi*nu*k)
    nu: normalised CFO  (fraction of symbol rate, typically ±0.1)
    phi0: initial phase offset.
    During training we randomise both to force the decoder to be robust.
    """
    def __init__(self, cfo: float = 0.05, phi0: float = 0.0,
                 cfo_range=(-0.15, 0.15), randomise: bool = True):
        super().__init__()
        self.cfo        = cfo
        self.phi0       = phi0
        self.cfo_range  = cfo_range
        self.randomise  = randomise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.randomise and self.training:
            lo, hi = self.cfo_range
            nu   = lo + torch.rand(1).item() * (hi - lo)
            phi0 = torch.rand(1).item() * 2 * np.pi
        else:
            nu   = self.cfo
            phi0 = self.phi0

        xc   = _to_complex(x)                      # (B, n)
        n    = xc.shape[1]
        k    = torch.arange(n, dtype=torch.float32, device=x.device)
        rot  = torch.exp(1j * (2 * np.pi * nu * k + phi0))
        yc   = xc * rot.unsqueeze(0)
        return _to_real(yc)


# ── 3. STO (Symbol Timing Offset) ────────────────────────────────────────────

class STOChannel(nn.Module):
    """
    Fractional timing offset implemented as a phase ramp in frequency domain
    (raised-cosine approximation for sub-sample shifts).

    For integer offsets we roll the symbol vector.  For fractional offsets
    we apply a linear phase in frequency (Fourier shift theorem).

    Here we use a simple soft version: additive ISI from adjacent symbols
    scaled by sin(pi*tau)/pi where tau ∈ (-0.5, 0.5) is the fractional offset.
    """
    def __init__(self, sto: float = 0.1, sto_range=(-0.4, 0.4),
                 randomise: bool = True):
        super().__init__()
        self.sto       = sto
        self.sto_range = sto_range
        self.randomise = randomise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.randomise and self.training:
            lo, hi = self.sto_range
            tau = lo + torch.rand(1).item() * (hi - lo)
        else:
            tau = self.sto

        # Sinc-weighted ISI from self (amplitude attenuation) + neighbour
        # For single-symbol blocks this reduces to a scalar attenuation + noise.
        sinc_val = np.sinc(tau)   # sin(pi*tau)/(pi*tau)
        leaked   = np.sqrt(max(0.0, 1.0 - sinc_val**2))

        xc      = _to_complex(x)
        noise_c = (torch.randn_like(xc) + 1j * torch.randn_like(xc)) * (leaked / np.sqrt(2))
        yc      = xc * sinc_val + noise_c
        return _to_real(yc)


# ── 4. Rayleigh Fading ────────────────────────────────────────────────────────

class RayleighChannel(nn.Module):
    """
    Flat (slow) Rayleigh fading: each sample in the batch gets an independent
    complex Gaussian fade coefficient.  Assumes channel is unknown to receiver.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = _to_complex(x)                    # (B, n)
        # Rayleigh: h ~ CN(0,1), normalise so E[|h|^2]=1
        h  = (torch.randn(xc.shape[0], 1, device=x.device) +
              1j * torch.randn(xc.shape[0], 1, device=x.device)) / np.sqrt(2)
        yc = xc * h
        return _to_real(yc)


# ── 5. Doppler (time-varying phase) ──────────────────────────────────────────

class DopplerChannel(nn.Module):
    """
    Time-varying Doppler modelled as a sinusoidal phase variation:
        phi(k) = A * sin(2*pi*fd*k / fs)
    fd: Doppler frequency (Hz), fs: symbol rate (Hz).
    Typical: fd/fs ~ 0.01 (pedestrian), 0.05 (vehicular).
    """
    def __init__(self, fd_norm: float = 0.02, amplitude: float = 1.0,
                 randomise: bool = True):
        super().__init__()
        self.fd_norm   = fd_norm
        self.amplitude = amplitude
        self.randomise = randomise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.randomise and self.training:
            fd = 0.005 + torch.rand(1).item() * 0.045
        else:
            fd = self.fd_norm

        xc  = _to_complex(x)
        n   = xc.shape[1]
        k   = torch.arange(n, dtype=torch.float32, device=x.device)
        phi = self.amplitude * torch.sin(2 * np.pi * fd * k)
        rot = torch.exp(1j * phi)
        yc  = xc * rot.unsqueeze(0)
        return _to_real(yc)


# ── 6. Impulsive Noise (non-Gaussian) ────────────────────────────────────────

class ImpulsiveNoiseChannel(nn.Module):
    """
    Bernoulli-Gaussian impulsive noise (Middleton Class-A model, simplified):
        n = (1-b)*g + b*(g + spike)
    b    : impulsive probability (~0.01–0.1)
    k_imp: spike amplitude relative to AWGN std
    """
    def __init__(self, snr_db: float = 10.0,
                 b: float = 0.05, k_imp: float = 10.0):
        super().__init__()
        self.snr_db = snr_db
        self.b      = b
        self.k_imp  = k_imp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std   = snr_to_noise_std(self.snr_db)
        awgn  = torch.randn_like(x) * std

        mask  = (torch.rand_like(x) < self.b).float()
        spike = torch.randn_like(x) * std * self.k_imp
        noise = awgn + mask * spike
        return x + noise


# ── 7. Colored Noise ─────────────────────────────────────────────────────────

class ColoredNoiseChannel(nn.Module):
    """
    First-order AR colored noise: n[k] = alpha*n[k-1] + sqrt(1-alpha^2)*w[k]
    alpha close to 1 → high correlation (pink-ish).
    """
    def __init__(self, snr_db: float = 10.0, alpha: float = 0.7):
        super().__init__()
        self.snr_db = snr_db
        self.alpha  = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std   = snr_to_noise_std(self.snr_db)
        B, D  = x.shape
        noise = torch.zeros_like(x)
        prev  = torch.zeros(B, device=x.device)
        scale = std * np.sqrt(1 - self.alpha**2)
        for i in range(D):
            w       = torch.randn(B, device=x.device) * scale
            prev    = self.alpha * prev + w
            noise[:, i] = prev
        return x + noise


# ── 8. Composite channel (training: randomised impairments) ──────────────────

class CompositeChannel(nn.Module):
    """
    Stacks AWGN + CFO + STO.  Used during autoencoder training so the network
    learns to handle all impairments jointly.
    snr_range : (lo, hi) dB — SNR sampled uniformly each batch.
    """
    def __init__(self, snr_range=(-4, 20)):
        super().__init__()
        self.awgn = AWGNChannel(snr_range=snr_range)
        self.cfo  = CFOChannel(randomise=True)
        self.sto  = STOChannel(randomise=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cfo(x)
        x = self.sto(x)
        x = self.awgn(x)
        return x

    def set_eval_snr(self, snr_db: float) -> None:
        """Fix SNR for deterministic evaluation."""
        self.awgn.snr_db    = snr_db
        self.awgn.snr_range = None
        self.cfo.randomise  = False
        self.sto.randomise  = False
