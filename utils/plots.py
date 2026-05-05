"""
plots.py — All publication-quality plots.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F

from utils   import DEVICE, ensure_dirs
from channel import AWGNChannel

PLOT_DIR = "plots"
DPI      = 150
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

def _save(fig, name):
    ensure_dirs(PLOT_DIR)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


def plot_ber_baseline(results):
    snr = np.array(results["snr_db"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr, np.clip(results["bpsk_theory"],  1e-5,1), "k-",  lw=2,   label="BPSK (theory)")
    ax.semilogy(snr, np.clip(results["qpsk_theory"],  1e-5,1), "b--", lw=2,   label="QPSK (theory)")
    ax.semilogy(snr, np.clip(results["qam16_theory"], 1e-5,1), "g:",  lw=2,   label="16-QAM (theory)")
    ax.semilogy(snr, np.clip(results["ae_awgn"],      1e-5,1), "r-o", lw=2, ms=5, label="AE – AWGN (pilotless)")
    ax.semilogy(snr, np.clip(results["ae_composite"], 1e-5,1), "m-s", lw=2, ms=5, label="AE – Composite (CFO+STO)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title("BER: Neural Autoencoder vs Classical Modulations")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(1e-5, 1); ax.set_xlim(snr[0], snr[-1])
    return _save(fig, "ber_baseline_vs_ae.png")


def plot_ber_coded(results):
    snr = np.array(results["snr_db"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr, np.clip(results["bpsk_theory"],   1e-5,1), "k-",  lw=1.5, label="BPSK (uncoded)")
    ax.semilogy(snr, np.clip(results["ldpc_approx"],   1e-5,1), "b-^", lw=2, ms=5, label="LDPC r=2/3 [APPROX]")
    ax.semilogy(snr, np.clip(results["turbo_approx"],  1e-5,1), "g-D", lw=2, ms=5, label="Turbo r=1/3 [APPROX]")
    ax.semilogy(snr, np.clip(results["polar_approx"],  1e-5,1), "c-v", lw=2, ms=5, label="Polar n=512 [APPROX]")
    ax.semilogy(snr, np.clip(results["ae_awgn"],       1e-5,1), "r-o", lw=2, ms=5, label="AE – AWGN (pilotless)")
    ax.semilogy(snr, np.clip(results["ae_composite"],  1e-5,1), "m-s", lw=2, ms=5, label="AE – Composite (pilotless)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title("BER vs Error-Correcting Codes\n(coded curves: published waterfall approximations)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(1e-5, 1); ax.set_xlim(snr[0], snr[-1])
    return _save(fig, "ber_coded_comparison.png")


def plot_ber_generalization(results):
    snr = np.array(results["snr_db"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr, np.clip(results["ae_awgn"],      1e-5,1), "r-o",  lw=2, ms=5, label="AE – AWGN (train dist.)")
    ax.semilogy(snr, np.clip(results["ae_composite"], 1e-5,1), "m-s",  lw=2, ms=5, label="AE – Composite (train dist.)")
    ax.semilogy(snr, np.clip(results["ae_rayleigh"],  1e-5,1), "b-^",  lw=2, ms=5, label="AE – Rayleigh (unseen)")
    ax.semilogy(snr, np.clip(results["ae_mismatch"],  1e-5,1), "g-D",  lw=2, ms=5, label="AE – Strong mismatch (unseen)")
    ax.semilogy(snr, np.clip(results["ae_doppler"],   1e-5,1), "c-v",  lw=2, ms=5, label="AE – Doppler (unseen)")
    ax.semilogy(snr, np.clip(results["bpsk_theory"],  1e-5,1), "k--",  lw=1.5,     label="BPSK (ref)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title("Generalisation: Trained on AWGN+Composite, Tested on Unseen Channels")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(1e-5, 1); ax.set_xlim(snr[0], snr[-1])
    return _save(fig, "ber_generalization.png")


def plot_ber_nongaussian(results):
    snr = np.array(results["snr_db"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr, np.clip(results["ae_awgn"],      1e-5,1), "r-o", lw=2, ms=5, label="AE – AWGN (Gaussian)")
    ax.semilogy(snr, np.clip(results["ae_impulsive"], 1e-5,1), "b-^", lw=2, ms=5, label="AE – Impulsive (b=0.05, k=10)")
    ax.semilogy(snr, np.clip(results["ae_colored"],   1e-5,1), "g-D", lw=2, ms=5, label="AE – Colored (α=0.7)")
    ax.semilogy(snr, np.clip(results["bpsk_theory"],  1e-5,1), "k--", lw=1.5,     label="BPSK AWGN (ref)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title("Non-Gaussian Noise Robustness")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(1e-5, 1); ax.set_xlim(snr[0], snr[-1])
    return _save(fig, "ber_nongaussian.png")


def plot_constellation(model, device=None, n_noise=2000):
    if device is None: device = DEVICE
    model.eval()
    M   = model.M
    pts = model.get_constellation(device)

    # Clean
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts[:,0], pts[:,1], c=range(M), cmap="tab20", s=120, zorder=5)
    for i,(x,y) in enumerate(pts):
        ax.annotate(str(i), (x,y), textcoords="offset points",
                    xytext=(5,3), fontsize=7, color="gray")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_title(f"Learned Constellation (M={M}, clean)"); ax.set_aspect("equal")
    p1 = _save(fig, "constellation_clean.png")

    # Impaired
    from channel import CFOChannel, STOChannel
    class FixedComposite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfo  = CFOChannel(cfo=0.05,  randomise=False)
            self.sto  = STOChannel(sto=0.1,   randomise=False)
            self.awgn = AWGNChannel(snr_db=10)
        def forward(self,x): return self.awgn(self.sto(self.cfo(x)))

    ch = FixedComposite().to(device); ch.eval()
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.tab20

    with torch.no_grad():
        for sym in range(M):
            lbl = torch.full((n_noise//M,), sym, dtype=torch.long, device=device)
            oh  = F.one_hot(lbl, M).float()
            sig = model.encoder(oh)
            rx  = ch(sig).cpu().numpy()
            ax.scatter(rx[:,0], rx[:,1], alpha=0.25, s=6,
                       c=[cmap(sym/M)]*len(rx))

    ax.scatter(pts[:,0], pts[:,1], c="black", s=60, zorder=5, marker="x",
               label="Clean constellation points")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("I (received)"); ax.set_ylabel("Q (received)")
    ax.set_title("Learned Constellation under Channel (CFO=0.05, STO=0.1, SNR=10dB)\n"
                 "×=clean points, scatter=received samples")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    p2 = _save(fig, "constellation_impaired.png")
    return p1, p2


def plot_training_curves(history):
    # Support both old format (epoch key) and new (phase key)
    n      = len(history["loss"])
    epochs = history.get("epoch", list(range(1, n+1)))
    loss   = history["loss"]
    acc    = history["acc"]
    phases = history.get("phase", ["train"]*n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    def smooth(y, w=30):
        k = np.ones(w)/w
        return np.convolve(y, k, mode="same")

    # Color by phase
    phase_colors = {"P1-AWGN": "steelblue", "P2-sync": "darkorange",
                    "P3-joint": "forestgreen", "train": "steelblue"}
    phase_labels = {"P1-AWGN": "Phase 1: AWGN (train all)",
                    "P2-sync": "Phase 2: Composite (freeze enc)",
                    "P3-joint": "Phase 3: Full joint",
                    "train": "Training"}

    for ax, data, ylabel, title in [
        (ax1, loss, "Cross-entropy loss", "Training Loss"),
        (ax2, acc,  "Symbol accuracy",    "Training Accuracy"),
    ]:
        ax.plot(epochs, data, alpha=0.2, color="gray", lw=0.6)
        ax.plot(epochs, smooth(data), color="steelblue" if ax==ax1 else "coral",
                lw=2, label="Smoothed")

        # Draw phase boundaries
        seen = set()
        for i, ph in enumerate(phases):
            if ph not in seen:
                seen.add(ph)
                if ph != phases[0]:
                    ax.axvline(x=epochs[i], color=phase_colors.get(ph,"gray"),
                               lw=1.5, ls="--", alpha=0.7,
                               label=phase_labels.get(ph, ph))

        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8)
        if ax == ax2: ax.set_ylim(0, 1.05)

    fig.suptitle("Autoencoder Training Curves (3-Phase)", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, "training_curves.png")


def generate_all_plots(model, results, history, device=None):
    paths = []
    print("\n  Generating plots...")
    paths.append(plot_ber_baseline(results))
    paths.append(plot_ber_coded(results))
    paths.append(plot_ber_generalization(results))
    paths.append(plot_ber_nongaussian(results))
    c1, c2 = plot_constellation(model, device)
    paths += [c1, c2]
    paths.append(plot_training_curves(history))
    print(f"  {len(paths)} plots saved to '{PLOT_DIR}/'")
    return paths


if __name__ == "__main__":
    import sys
    from model import AutoEncoder
    from train import DEFAULT_CFG
    save_dir = "results"
    model_pt = os.path.join(save_dir, "best_model.pt")
    if not os.path.exists(model_pt):
        print("Run train.py first."); sys.exit(1)
    model = AutoEncoder(DEFAULT_CFG["M"], DEFAULT_CFG["n_channel"],
                        DEFAULT_CFG["hidden"]).to(DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE))
    with open(os.path.join(save_dir, "ber_results.json")) as f:
        results = json.load(f)
    with open(os.path.join(save_dir, "history.json")) as f:
        history = json.load(f)
    generate_all_plots(model, results, history)
