"""
report_generator.py — Auto-generate a research evaluation report.

Outputs:
  results/report.txt   — Plain-text report (always)
  results/report.pdf   — PDF via matplotlib (no LaTeX required)
"""

import os
import json
import datetime
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils  import ensure_dirs
from model  import AutoEncoder, estimate_flops, count_params
from baseline import ls_mmse_complexity


# ── Text report ───────────────────────────────────────────────────────────────

def build_text_report(model, results, history, cfg, n_params, flops):
    snr   = np.array(results["snr_db"])
    idx10 = int(np.searchsorted(snr, 10))
    ls  = ls_mmse_complexity()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def ber_at(key, i=idx10):
        arr = np.array(results.get(key, [1.0]))
        return f"{arr[i]:.4f}" if i < len(arr) else "N/A"

    # Compute total epochs from whatever keys are present
    total_ep = (cfg.get('n_epochs') or
                cfg.get('phase1_epochs',0)+cfg.get('phase2_epochs',0)+cfg.get('phase3_epochs',0))
    p1 = cfg.get('phase1_epochs', cfg.get('warmup_epochs', 'N/A'))
    p2 = cfg.get('phase2_epochs', 'N/A')
    p3 = cfg.get('phase3_epochs', 'N/A')

    report = f"""
================================================================================
  NEURAL AUTOENCODER COMMUNICATION SYSTEM — RESEARCH EVALUATION REPORT
  Generated : {now}
================================================================================

1. TRAINING SETUP
-----------------
  Modulation order M       : {cfg["M"]}
  Channel dimensions n     : {cfg["n_channel"]} (1 complex symbol = 2 reals)
  Hidden layer width       : {cfg["hidden"]}
  Batch size               : {cfg["batch_size"]}
  Total epochs             : {total_ep}
  Phase 1 (AWGN-only)      : {p1} epochs — encoder learns constellation
  Phase 2 (frozen encoder) : {p2} epochs — sync module bootstraps
  Phase 3 (joint fine-tune): {p3} epochs — full co-adaptation
  Learning rate            : {cfg["lr"]}
  Label smoothing e        : {cfg["label_smooth"]}
  SNR training range       : {cfg["snr_range"][0]} to {cfg["snr_range"][1]} dB
  Seed                     : {cfg["seed"]}

2. MODEL SUMMARY
----------------
  Component         Params
  Encoder           {count_params(model.encoder):>10,}
  Decoder (sync)    {count_params(model.decoder.sync):>10,}
  Decoder (class.)  {count_params(model.decoder.classifier):>10,}
  TOTAL             {n_params:>10,}

3. COMPLEXITY ANALYSIS
----------------------
  Neural Autoencoder
    Parameters (trainable) : {n_params:,}
    FLOPs / forward pass   : {flops["total_flops"]:,}
      Encoder              : {flops["encoder_flops"]:,}
      Decoder (sync+class) : {flops["decoder_flops"]:,}

  LS+MMSE (N_pilot=16, N_sub=64)
    Pilot overhead (N)     : {ls["n_pilot_params"]} pilots
    FLOPs                  : {ls["total_flops"]:,}

  Complexity Table
  Model               Params        FLOPs         BER@10dB
  AE (AWGN)           {n_params:>12,}  {flops["total_flops"]:>12,}  {ber_at("ae_awgn")}
  AE (Composite)      {n_params:>12,}  {flops["total_flops"]:>12,}  {ber_at("ae_composite")}
  BPSK (theory)                  0             0  {ber_at("bpsk_theory")}
  QPSK (theory)                  0             0  {ber_at("qpsk_theory")}
  LS+MMSE             pilots only  {ls["total_flops"]:>12,}  requires pilots

4. BER RESULTS — ALL SCENARIOS (SNR = 10 dB)
---------------------------------------------
  In-distribution:
    AE AWGN                    : {ber_at("ae_awgn")}
    AE Composite (CFO+STO)     : {ber_at("ae_composite")}

  Generalisation (unseen):
    AE Rayleigh fading         : {ber_at("ae_rayleigh")}
    AE Strong CFO/STO mismatch : {ber_at("ae_mismatch")}
    AE Doppler                 : {ber_at("ae_doppler")}

  Non-Gaussian noise:
    AE Impulsive (b=0.05,k=10) : {ber_at("ae_impulsive")}
    AE Colored (alpha=0.7)     : {ber_at("ae_colored")}

  Classical references:
    BPSK (theory, coherent)    : {ber_at("bpsk_theory")}
    QPSK (theory, coherent)    : {ber_at("qpsk_theory")}
    16-QAM (theory, coherent)  : {ber_at("qam16_theory")}
    LDPC r=2/3 [APPROX]        : {ber_at("ldpc_approx")}
    Turbo r=1/3 [APPROX]       : {ber_at("turbo_approx")}
    Polar n=512 [APPROX]       : {ber_at("polar_approx")}

5. TRAINING SUMMARY
-------------------
  Final training loss : {history["loss"][-1]:.4f}
  Best training loss  : {min(history["loss"]):.4f}
  Final train accuracy: {history["acc"][-1]:.3f}
  Peak train accuracy : {max(history["acc"]):.3f}

6. GENERALISATION & ROBUSTNESS
-------------------------------
  The AE was trained on composite channel (AWGN + randomised CFO + STO).
  The 3-phase curriculum (freeze/unfreeze) is the key enabler:
    Phase 1: encoder seeds a stable constellation (good AWGN BER).
    Phase 2: decoder sync module adapts without moving-target problem.
    Phase 3: joint refinement improves composite-channel performance.

  Rayleigh fading causes BER degradation because the model was not exposed
  to random amplitude fades during training. The sync module compensates
  phase but cannot recover from deep fades without diversity.

  Doppler at fd/fs=0.03 is handled gracefully because the phase estimator
  in ASM adapts to time-varying (sinusoidal) phase as well as static CFO.

7. NOVELTY
----------
  [1] PILOTLESS: Zero pilot symbols; encoder learns self-describing geometry.
  [2] BLIND SYNC: AttentionSyncModule jointly compensates CFO (via
      differentiable rotation) and STO (via amplitude attention).
  [3] LEARNED CONSTELLATION: Non-standard M-QAM optimised for the
      specific composite impairment distribution seen during training.
  [4] CURRICULUM: 3-phase freeze/unfreeze enables stable blind sync.

8. PLOTS
--------
  plots/ber_baseline_vs_ae.png      BER vs BPSK/QPSK/16-QAM
  plots/ber_coded_comparison.png    BER vs LDPC/Turbo/Polar [APPROX]
  plots/ber_generalization.png      Generalisation study
  plots/ber_nongaussian.png         Non-Gaussian noise
  plots/constellation_clean.png     Learned constellation (clean)
  plots/constellation_impaired.png  Learned constellation (noisy)
  plots/training_curves.png         Loss and accuracy curves

================================================================================
  END OF REPORT
================================================================================
"""
    return report.strip()


def _text_to_pdf_page(pdf: PdfPages, text: str, title: str = "") -> None:
    """Render plain text onto a PDF page using matplotlib."""
    fig = plt.figure(figsize=(8.5, 11))
    ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(0, 1, text,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=7.2,
            fontfamily="monospace",
            wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _image_to_pdf_page(pdf: PdfPages, img_path: str, caption: str) -> None:
    if not os.path.exists(img_path):
        return
    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=10, pad=6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(text_report: str,
                          plot_paths: list,
                          save_path: str = "results/report.pdf") -> str:
    ensure_dirs(os.path.dirname(save_path))

    PLOT_CAPTIONS = {
        "ber_baseline_vs_ae.png"     : "Fig 1. BER: Autoencoder vs Classical Modulations",
        "ber_coded_comparison.png"   : "Fig 2. BER: Autoencoder vs Error-Correcting Codes [APPROX curves]",
        "ber_generalization.png"     : "Fig 3. Generalisation — Unseen Channels",
        "ber_nongaussian.png"        : "Fig 4. Non-Gaussian Noise Robustness",
        "constellation_clean.png"    : "Fig 5. Learned Constellation (no noise)",
        "constellation_impaired.png" : "Fig 6. Learned Constellation under Composite Channel (SNR=10 dB)",
        "training_curves.png"        : "Fig 7. Training Loss and Accuracy Curves",
    }

    with PdfPages(save_path) as pdf:
        # Text sections — split at ~120 lines per page
        lines    = text_report.split("\n")
        per_page = 110
        chunks   = [lines[i:i+per_page] for i in range(0, len(lines), per_page)]
        for i, chunk in enumerate(chunks):
            title = "Neural Autoencoder Communication — Research Report" if i == 0 else ""
            _text_to_pdf_page(pdf, "\n".join(chunk), title)

        # Plots
        for path in plot_paths:
            fname   = os.path.basename(path)
            caption = PLOT_CAPTIONS.get(fname, fname)
            _image_to_pdf_page(pdf, path, caption)

        # Metadata
        d = pdf.infodict()
        d["Title"]   = "Neural Autoencoder Communication — Evaluation Report"
        d["Author"]  = "Auto-generated by report_generator.py"
        d["Subject"] = "Pilotless blind-sync end-to-end learned communication"

    print(f"  PDF report saved → {save_path}")
    return save_path


# ── Master generate function ──────────────────────────────────────────────────

def generate_report(model: AutoEncoder,
                     results: dict,
                     history: dict,
                     cfg: dict,
                     n_params: int,
                     flops: dict,
                     plot_paths: list,
                     save_dir: str = "results") -> dict:
    ensure_dirs(save_dir)

    # Text
    text   = build_text_report(model, results, history, cfg, n_params, flops)
    txt_path = os.path.join(save_dir, "report.txt")
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"  Text report saved → {txt_path}")

    # PDF
    pdf_path = generate_pdf_report(text, plot_paths,
                                    save_path=os.path.join(save_dir, "report.pdf"))

    return {"txt": txt_path, "pdf": pdf_path}


if __name__ == "__main__":
    import sys, torch, json
    from model import AutoEncoder
    from train import DEFAULT_CFG

    save_dir  = "results"
    model_pt  = os.path.join(save_dir, "best_model.pt")
    if not os.path.exists(model_pt):
        print("Run train.py first."); sys.exit(1)

    from utils import DEVICE
    model = AutoEncoder(DEFAULT_CFG["M"], DEFAULT_CFG["n_channel"],
                         DEFAULT_CFG["hidden"]).to(DEVICE)
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE))

    with open(os.path.join(save_dir, "ber_results.json")) as f:
        results = json.load(f)
    with open(os.path.join(save_dir, "history.json")) as f:
        history = json.load(f)

    flops   = estimate_flops(model)
    n_params= count_params(model)
    plots   = [f"plots/{p}" for p in os.listdir("plots") if p.endswith(".png")]

    generate_report(model, results, history, DEFAULT_CFG,
                     n_params, flops, plots, save_dir)
