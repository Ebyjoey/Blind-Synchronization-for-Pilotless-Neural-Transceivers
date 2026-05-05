"""
evaluation.py — BER evaluation for all scenarios.

Scenarios evaluated
-------------------
  1. AWGN only (in-distribution)
  2. Composite (AWGN + CFO + STO)  — training distribution
  3. Rayleigh fading                — generalisation
  4. CFO/STO mismatch               — generalisation (stronger impairment)
  5. Doppler                        — generalisation
  6. Impulsive noise                — non-Gaussian
  7. Colored noise                  — non-Gaussian

All evaluated by Monte-Carlo over n_symbols per SNR point.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from utils    import DEVICE, set_seed, ensure_dirs
from model    import AutoEncoder
from channel  import (AWGNChannel, CompositeChannel, RayleighChannel,
                       CFOChannel, STOChannel, DopplerChannel,
                       ImpulsiveNoiseChannel, ColoredNoiseChannel)
from baseline import (ber_bpsk_theory, ber_qpsk_theory, ber_16qam_theory,
                       ber_ldpc_approx, ber_turbo_approx, ber_polar_approx,
                       simulate_bpsk, simulate_qpsk)


# ── Core BER evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def eval_ber(model: AutoEncoder, channel_fn,
             snr_db_list, n_symbols: int = 50_000,
             device=None) -> np.ndarray:
    """
    Evaluate BER of the autoencoder over a list of SNR values.

    The channel is given a fixed SNR by patching its attributes directly
    so we can reuse the same channel object without re-instantiation.
    """
    if device is None:
        device = DEVICE
    model.eval()
    M      = model.M
    bers   = []

    for snr_db in snr_db_list:
        # Fix SNR on any AWGN-containing channel
        _set_channel_snr(channel_fn, snr_db)

        # Generate symbols in batches
        batch   = 2048
        n_err   = 0
        n_bits  = 0
        n_done  = 0

        while n_done < n_symbols:
            bs      = min(batch, n_symbols - n_done)
            labels  = torch.randint(0, M, (bs,), device=device)
            one_hot = F.one_hot(labels, num_classes=M).float()

            signal  = model.encoder(one_hot)
            received= channel_fn(signal)
            logits  = model.decoder(received)
            preds   = logits.argmax(dim=-1)

            # Symbol errors → bit errors (Gray code approx: log2(M) bits/symbol)
            sym_err  = (preds != labels).sum().item()
            bits_sym = int(np.log2(M))
            n_err   += sym_err * bits_sym    # worst-case; ok for comparison
            n_bits  += bs * bits_sym
            n_done  += bs

        bers.append(n_err / max(n_bits, 1))

    return np.array(bers)


def _set_channel_snr(ch, snr_db: float) -> None:
    """Recursively set snr_db on channel and sub-channels, disable randomise."""
    for attr in ["snr_db", "snr_range"]:
        if hasattr(ch, attr):
            if attr == "snr_db":
                setattr(ch, attr, snr_db)
            else:
                setattr(ch, attr, None)
    for attr in ["randomise"]:
        if hasattr(ch, attr):
            setattr(ch, attr, False)
    # Recurse into sub-channels
    for sub in ["awgn", "cfo", "sto"]:
        if hasattr(ch, sub):
            _set_channel_snr(getattr(ch, sub), snr_db)


# ── Full evaluation suite ─────────────────────────────────────────────────────

def run_all_evaluations(model: AutoEncoder,
                         snr_range=(-4, 20),
                         snr_step=2,
                         n_symbols: int = 50_000,
                         save_dir: str = "results",
                         device=None) -> dict:
    if device is None:
        device = DEVICE
    ensure_dirs(save_dir)
    set_seed(0)

    snr_db_list = np.arange(snr_range[0], snr_range[1] + 1, snr_step,
                             dtype=float)

    print("\n  Running BER evaluations...")

    results = {"snr_db": snr_db_list.tolist()}

    # 1. Autoencoder — AWGN
    print("  [1/7] AE AWGN...")
    ch = AWGNChannel().to(device)
    ch.eval()
    results["ae_awgn"] = eval_ber(model, ch, snr_db_list, n_symbols, device).tolist()

    # 2. Autoencoder — Composite (training distribution)
    print("  [2/7] AE Composite (AWGN+CFO+STO)...")
    ch = CompositeChannel().to(device)
    ch.eval()
    results["ae_composite"] = eval_ber(model, ch, snr_db_list, n_symbols, device).tolist()

    # 3. Autoencoder — Rayleigh
    print("  [3/7] AE Rayleigh (unseen)...")
    def rayleigh_awgn(x):
        x = RayleighChannel().to(device)(x)
        return AWGNChannel(snr_db=10.0).to(device)(x)
    # We need to be able to set snr; use a wrapper class
    class RayleighAWGN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.awgn    = AWGNChannel()
            self.rayleigh= RayleighChannel()
        def forward(self, x):
            return self.awgn(self.rayleigh(x))
    ch = RayleighAWGN().to(device)
    ch.eval()
    results["ae_rayleigh"] = eval_ber(model, ch, snr_db_list, n_symbols, device).tolist()

    # 4. Autoencoder — Strong CFO/STO mismatch
    print("  [4/7] AE Strong CFO/STO mismatch...")
    class StrongMismatch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfo  = CFOChannel(cfo=0.20, randomise=False)
            self.sto  = STOChannel(sto=0.45, randomise=False)
            self.awgn = AWGNChannel()
        def forward(self, x):
            return self.awgn(self.sto(self.cfo(x)))
    ch = StrongMismatch().to(device)
    ch.eval()
    results["ae_mismatch"] = eval_ber(model, ch, snr_db_list, n_symbols, device).tolist()

    # 5. Autoencoder — Doppler
    print("  [5/7] AE Doppler...")
    class DopplerAWGN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.doppler = DopplerChannel(fd_norm=0.03, randomise=False)
            self.awgn    = AWGNChannel()
        def forward(self, x):
            return self.awgn(self.doppler(x))
    ch = DopplerAWGN().to(device)
    ch.eval()
    results["ae_doppler"] = eval_ber(model, ch, snr_db_list, n_symbols, device).tolist()

    # 6. Autoencoder — Impulsive noise
    print("  [6/7] AE Impulsive noise...")
    ch = ImpulsiveNoiseChannel(b=0.05, k_imp=10.0).to(device)
    ch.eval()
    results["ae_impulsive"] = eval_ber(model, ch, snr_db_list, n_symbols, device).tolist()

    # 7. Autoencoder — Colored noise
    print("  [7/7] AE Colored noise...")
    ch = ColoredNoiseChannel(alpha=0.7).to(device)
    ch.eval()
    results["ae_colored"] = eval_ber(model, ch, snr_db_list, n_symbols, device).tolist()

    # Classical baselines (theory + simulation)
    print("  Computing classical baselines...")
    results["bpsk_theory"]  = ber_bpsk_theory(snr_db_list).tolist()
    results["qpsk_theory"]  = ber_qpsk_theory(snr_db_list).tolist()
    results["qam16_theory"] = ber_16qam_theory(snr_db_list).tolist()
    results["bpsk_sim"]     = simulate_bpsk(snr_db_list, n_bits=50_000).tolist()
    results["ldpc_approx"]  = ber_ldpc_approx(snr_db_list).tolist()
    results["turbo_approx"] = ber_turbo_approx(snr_db_list).tolist()
    results["polar_approx"] = ber_polar_approx(snr_db_list).tolist()

    # Save
    out_path = os.path.join(save_dir, "ber_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")

    return results


if __name__ == "__main__":
    import sys
    save_dir  = "results"
    model_pt  = os.path.join(save_dir, "best_model.pt")

    if not os.path.exists(model_pt):
        print("ERROR: No trained model found. Run train.py first.")
        sys.exit(1)

    from train import DEFAULT_CFG
    model = AutoEncoder(DEFAULT_CFG["M"], DEFAULT_CFG["n_channel"],
                         DEFAULT_CFG["hidden"]).to(DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE))

    results = run_all_evaluations(model, save_dir=save_dir)
    print("\nBER @ 10 dB SNR:")
    for k, v in results.items():
        if k == "snr_db":
            continue
        arr = np.array(v)
        idx = np.searchsorted(results["snr_db"], 10)
        if idx < len(arr):
            print(f"  {k:20s}: {arr[idx]:.4f}")
