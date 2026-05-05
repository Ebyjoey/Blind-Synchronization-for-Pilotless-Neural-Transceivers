"""
model.py — Neural autoencoder with robust blind sync (fixed BER).

Key fix: stronger AttentionSyncModule with residual IQ normalization,
and a dedicated amplitude estimator that helps under both CFO and STO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import count_params, DEVICE


class PowerNorm(nn.Module):
    def forward(self, x):
        power = x.pow(2).sum(dim=-1, keepdim=True).mean().sqrt()
        return x / (power + 1e-8)


class Encoder(nn.Module):
    def __init__(self, M=16, n_channel=2, hidden=256):
        super().__init__()
        self.M = M
        self.n_channel = n_channel
        self.net = nn.Sequential(
            nn.Linear(M, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Linear(hidden, n_channel),
        )
        self.power_norm = PowerNorm()

    def forward(self, x):
        return self.power_norm(self.net(x))


class AttentionSyncModule(nn.Module):
    """
    Robust blind sync front-end.
    
    Architecture:
      1. Amplitude normaliser: remove unknown STO-induced gain
         (learn to divide by estimated amplitude)
      2. Phase estimator: MLP → delta_phi, apply rotation
      3. Residual attention: re-weight IQ components after rotation
    
    The amplitude normalization is the critical addition — it decouples
    the phase estimation from gain estimation, making each easier.
    """
    def __init__(self, n_channel=2, hidden=128):
        super().__init__()

        # Step 1: amplitude estimation (STO compensation)
        self.amp_est = nn.Sequential(
            nn.Linear(n_channel, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),    # positive output = estimated amplitude
        )

        # Step 2: phase estimation (CFO compensation)
        self.phase_net = nn.Sequential(
            nn.Linear(n_channel, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 2),   # [cos_phi, sin_phi] — avoids angle wrapping
        )

        # Step 3: residual feature mixer
        self.mixer = nn.Sequential(
            nn.Linear(n_channel, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, n_channel),
        )

    def forward(self, y):
        # 1. Amplitude normalisation
        amp    = self.amp_est(y) + 1e-3      # (B, 1)
        y_norm = y / amp                      # (B, 2)

        # 2. Phase rotation using cos/sin parameterisation
        cs     = self.phase_net(y_norm)       # (B, 2): [cos_hat, sin_hat]
        cos_p  = cs[:, 0:1]
        sin_p  = cs[:, 1:2]
        # Normalise to unit circle (ensures valid rotation)
        norm   = torch.sqrt(cos_p**2 + sin_p**2 + 1e-8)
        cos_p  = cos_p / norm
        sin_p  = sin_p / norm

        I_rot  = cos_p * y_norm[:, 0:1] - sin_p * y_norm[:, 1:2]
        Q_rot  = sin_p * y_norm[:, 0:1] + cos_p * y_norm[:, 1:2]
        y_rot  = torch.cat([I_rot, Q_rot], dim=-1)   # (B, 2)

        # 3. Residual mixing
        y_out  = y_rot + self.mixer(y_rot)
        return y_out


class Decoder(nn.Module):
    def __init__(self, M=16, n_channel=2, hidden=256):
        super().__init__()
        self.M    = M
        self.sync = AttentionSyncModule(n_channel, hidden=128)
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, M),
        )

    def forward(self, y):
        return self.classifier(self.sync(y))


class AutoEncoder(nn.Module):
    def __init__(self, M=16, n_channel=2, hidden=256):
        super().__init__()
        self.M         = M
        self.n_channel = n_channel
        self.encoder   = Encoder(M, n_channel, hidden)
        self.decoder   = Decoder(M, n_channel, hidden)

    def forward(self, x_onehot, channel):
        s      = self.encoder(x_onehot)
        y      = channel(s)
        return self.decoder(y)

    def get_constellation(self, device=None):
        if device is None:
            device = next(self.parameters()).device
        eye = torch.eye(self.M, device=device)
        with torch.no_grad():
            return self.encoder(eye).cpu().numpy()

    def summary(self):
        enc_p = count_params(self.encoder)
        dec_p = count_params(self.decoder)
        return {"encoder_params": enc_p, "decoder_params": dec_p,
                "total_params": enc_p + dec_p}


def estimate_flops(model, batch_size=1):
    M, n = model.M, model.n_channel
    H, Hs = 256, 128

    def fc(a, b): return 2 * a * b

    enc  = fc(M, H) + fc(H, H) + fc(H, n)
    sync = (fc(n, Hs//2) + fc(Hs//2, 1) +        # amp_est
            fc(n, Hs) + fc(Hs, Hs//2) + fc(Hs//2, 2) +  # phase_net
            fc(n, Hs//2) + fc(Hs//2, n))           # mixer
    cls  = fc(n, H) + fc(H, H) + fc(H, H//2) + fc(H//2, M)
    dec  = sync + cls
    return {"encoder_flops": enc, "decoder_flops": dec,
            "total_flops": (enc + dec) * batch_size}
