"""
baseline.py — Classical communication baselines.

Provides:
  1. BPSK / QPSK coherent receivers (AWGN)
  2. Theoretical BER curves (BPSK, QPSK, 16-QAM)
  3. Approximate BER for LDPC, Turbo, Polar codes
     (Based on published waterfall curves; marked as approximations.)
  4. LS + MMSE complexity estimate

Note on LDPC/Turbo/Polar
-------------------------
Full codec implementation is out of scope for a single research demo.
We use published BER curve approximations parameterised as sigmoid-shaped
waterfall functions fitted to standard references:
  - LDPC  : IEEE 802.11n (648, 432) rate-2/3
  - Turbo : LTE rate-1/3, K=40
  - Polar : 5G NR (512, 256) CA-SCL-8
These are clearly labelled as approximations in all plots and the report.
"""

import numpy as np
from scipy.special import erfc
from utils import snr_to_noise_std, compute_ber


# ── Theoretical BER ───────────────────────────────────────────────────────────

def ber_bpsk_theory(snr_db: np.ndarray) -> np.ndarray:
    """BER = Q(sqrt(2*Eb/N0)) = 0.5*erfc(sqrt(Eb/N0))"""
    ebn0 = 10.0 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0))


def ber_qpsk_theory(snr_db: np.ndarray) -> np.ndarray:
    """QPSK same as BPSK per bit in AWGN."""
    return ber_bpsk_theory(snr_db)


def ber_16qam_theory(snr_db: np.ndarray) -> np.ndarray:
    """Approximate Gray-coded 16-QAM BER."""
    ebn0 = 10.0 ** (snr_db / 10.0)
    # Es/N0 = 4 * Eb/N0 for 16-QAM (4 bits/symbol)
    ber  = (3.0 / 8.0) * erfc(np.sqrt(ebn0 * 4.0 / 10.0))
    return ber


# ── Coded BER approximations ──────────────────────────────────────────────────
# Sigmoid waterfall: BER ≈ 1/(1+exp(a*(SNR-SNR_th)))

def _waterfall(snr_db: np.ndarray, snr_th: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(slope * (snr_db - snr_th)))


def ber_ldpc_approx(snr_db: np.ndarray) -> np.ndarray:
    """
    Approximate BER for LDPC (rate 2/3, n=648) — IEEE 802.11n reference.
    [APPROX] Waterfall around ~3 dB Eb/N0.
    """
    raw = _waterfall(snr_db, snr_th=2.8, slope=3.5)
    return np.clip(raw * 0.5, 1e-6, 0.5)


def ber_turbo_approx(snr_db: np.ndarray) -> np.ndarray:
    """
    Approximate BER for Turbo (LTE, rate 1/3, K=40) — 3GPP reference.
    [APPROX] Waterfall around ~1.5 dB Eb/N0.
    """
    raw = _waterfall(snr_db, snr_th=1.2, slope=4.0)
    return np.clip(raw * 0.5, 1e-6, 0.5)


def ber_polar_approx(snr_db: np.ndarray) -> np.ndarray:
    """
    Approximate BER for Polar (5G NR, n=512, k=256, CA-SCL-8) — 3GPP ref.
    [APPROX] Waterfall around ~2.5 dB Eb/N0.
    """
    raw = _waterfall(snr_db, snr_th=2.2, slope=4.5)
    return np.clip(raw * 0.5, 1e-6, 0.5)


# ── Simulated BPSK / QPSK receivers ──────────────────────────────────────────

def simulate_bpsk(snr_db_list, n_bits: int = 100_000) -> np.ndarray:
    """
    Coherent BPSK detector over AWGN.
    Returns empirical BER at each SNR point.
    """
    bers = []
    for snr_db in snr_db_list:
        std   = snr_to_noise_std(snr_db, rate=1.0)
        bits  = np.random.randint(0, 2, n_bits)
        syms  = 2 * bits - 1                      # ±1
        noise = np.random.randn(n_bits) * std
        rx    = syms + noise
        bits_hat = (rx > 0).astype(int)
        bers.append(compute_ber(bits, bits_hat))
    return np.array(bers)


def simulate_qpsk(snr_db_list, n_bits: int = 100_000) -> np.ndarray:
    """
    Coherent QPSK detector over AWGN.
    2 bits per symbol, Gray-coded.
    """
    bers = []
    for snr_db in snr_db_list:
        n_syms = n_bits // 2
        std    = snr_to_noise_std(snr_db, rate=2.0)

        bits   = np.random.randint(0, 2, (n_syms, 2))
        # Gray-coded QPSK: (b0,b1) → (±1/√2 + j*±1/√2)
        I = (2 * bits[:, 0] - 1) / np.sqrt(2)
        Q = (2 * bits[:, 1] - 1) / np.sqrt(2)

        nI = np.random.randn(n_syms) * std
        nQ = np.random.randn(n_syms) * std

        bits_hat = np.stack([(I + nI > 0).astype(int),
                              (Q + nQ > 0).astype(int)], axis=1)
        bers.append(compute_ber(bits.ravel(), bits_hat.ravel()))
    return np.array(bers)


# ── LS + MMSE complexity ──────────────────────────────────────────────────────

def ls_mmse_complexity(N_pilot: int = 16, N_data: int = 48,
                       N_sub: int = 64) -> dict:
    """
    FLOPs estimate for LS channel estimation + MMSE equalisation.
    N_pilot : pilot subcarriers
    N_data  : data subcarriers
    N_sub   : total subcarriers (FFT size)

    LS estimation  : O(N_pilot) complex divisions  → 6*N_pilot real flops
    MMSE filter    : O(N_sub^2) matrix inversion   → 2*N_sub^3 real flops (approx)
    Equalisation   : O(N_data) complex multiplies  → 6*N_data real flops
    """
    ls_flops   = 6 * N_pilot
    mmse_flops = 2 * N_sub**3     # dominant: matrix inversion
    eq_flops   = 6 * N_data
    total      = ls_flops + mmse_flops + eq_flops
    return {
        "ls_flops"  : ls_flops,
        "mmse_flops": mmse_flops,
        "eq_flops"  : eq_flops,
        "total_flops": total,
        "n_pilot_params": N_pilot,   # pilots = overhead, not learnable params
    }
