"""
train.py — 3-phase training for blind sync autoencoder.

Phase 1: Train encoder + decoder on AWGN only (high SNR).
         Encoder learns a clean M-point constellation.
         Decoder learns ideal detection.

Phase 2: FREEZE encoder. Train ONLY the decoder (sync + classifier)
         on composite channel. The decoder adapts its sync module
         without disturbing the learned constellation.

Phase 3: UNFREEZE all. Joint fine-tuning on composite channel
         with lower LR. Both encoder and decoder co-adapt.

This phased freeze/unfreeze is the correct recipe for blind sync:
- Phase 1 gives the sync module a stable target (known constellation).
- Phase 2 lets sync bootstrap without the moving-target problem.
- Phase 3 allows joint optimisation once sync is already functional.
"""

import os, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils   import set_seed, DEVICE, count_params, ensure_dirs
from model   import AutoEncoder, estimate_flops
from channel import AWGNChannel, CFOChannel, STOChannel


DEFAULT_CFG = dict(
    M             = 16,
    n_channel     = 2,
    hidden        = 256,
    batch_size    = 2048,
    # Phase epoch counts
    phase1_epochs = 200,   # AWGN, train all
    phase2_epochs = 300,   # composite, freeze encoder
    phase3_epochs = 200,   # composite, unfreeze all (joint)
    lr            = 3e-3,
    lr_phase3     = 5e-4,
    weight_decay  = 1e-5,
    label_smooth  = 0.05,
    snr_range     = (0, 20),
    seed          = 42,
    save_dir      = "results",
)


class MildComposite(torch.nn.Module):
    """Stage-2 mild impairments."""
    def __init__(self, snr_range=(2, 20)):
        super().__init__()
        self.awgn = AWGNChannel(snr_range=snr_range)
        self.cfo  = CFOChannel(cfo_range=(-0.08, 0.08), randomise=True)
        self.sto  = STOChannel(sto_range=(-0.2,  0.2),  randomise=True)
    def forward(self, x):
        return self.awgn(self.sto(self.cfo(x)))


class FullComposite(torch.nn.Module):
    """Stage-3 full impairments."""
    def __init__(self, snr_range=(0, 20)):
        super().__init__()
        self.awgn = AWGNChannel(snr_range=snr_range)
        self.cfo  = CFOChannel(cfo_range=(-0.15, 0.15), randomise=True)
        self.sto  = STOChannel(sto_range=(-0.4,  0.4),  randomise=True)
    def forward(self, x):
        return self.awgn(self.sto(self.cfo(x)))


def sample_batch(M, bs, device):
    labels  = torch.randint(0, M, (bs,), device=device)
    return labels, F.one_hot(labels, M).float()


def smooth_loss(logits, labels, eps=0.05):
    lp = F.log_softmax(logits, dim=-1)
    nll = -lp.gather(1, labels.unsqueeze(1)).squeeze(1)
    return ((1-eps)*nll + eps*(-lp.mean(dim=-1))).mean()


def _run_phase(model, channel, n_epochs, opt, sched, cfg,
               phase_name, device, history, best_loss, verbose):
    M, bs = cfg["M"], cfg["batch_size"]
    t0    = time.time()

    for ep in range(1, n_epochs + 1):
        model.train()
        labels, oh = sample_batch(M, bs, device)
        logits     = model(oh, channel)
        loss       = smooth_loss(logits, labels, cfg["label_smooth"])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched: sched.step()

        with torch.no_grad():
            acc = (logits.argmax(-1) == labels).float().mean().item()

        history["loss"].append(loss.item())
        history["acc"].append(acc)
        history["phase"].append(phase_name)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(),
                       os.path.join(cfg["save_dir"], "best_model.pt"))

        if verbose and ep % 100 == 0:
            tot_ep = len(history["loss"])
            print(f"  [{phase_name}] ep {ep:4d}/{n_epochs}  "
                  f"loss={loss.item():.4f}  acc={acc:.3f}  "
                  f"t={time.time()-t0:.1f}s")

    return best_loss


def train(cfg=None, verbose=True):
    if cfg is None: cfg = DEFAULT_CFG
    cfg = {**DEFAULT_CFG, **cfg}
    set_seed(cfg["seed"])
    ensure_dirs(cfg["save_dir"])
    device = DEVICE

    model    = AutoEncoder(cfg["M"], cfg["n_channel"], cfg["hidden"]).to(device)
    n_params = count_params(model)
    flops    = estimate_flops(model)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Neural AE — 3-Phase Blind Sync Training | {device}")
        print(f"  M={cfg['M']}  params={n_params:,}  FLOPs={flops['total_flops']:,}")
        print(f"  Phase1={cfg['phase1_epochs']}ep (AWGN) "
              f"Phase2={cfg['phase2_epochs']}ep (freeze-enc) "
              f"Phase3={cfg['phase3_epochs']}ep (joint)")
        print(f"{'='*60}")

    history   = {"loss": [], "acc": [], "phase": []}
    best_loss = float("inf")

    # ── Phase 1: AWGN, train all ──────────────────────────────────────────
    ch1 = AWGNChannel(snr_range=(4, 20)).to(device)
    opt1 = torch.optim.AdamW(model.parameters(),
                              lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch1 = CosineAnnealingWarmRestarts(opt1, T_0=100)

    if verbose: print(f"\n  Phase 1: AWGN-only (train all params)")
    best_loss = _run_phase(model, ch1, cfg["phase1_epochs"], opt1, sch1,
                            cfg, "P1-AWGN", device, history, best_loss, verbose)
    # Save phase-1 encoder separately so we can reload if needed
    torch.save(model.encoder.state_dict(),
               os.path.join(cfg["save_dir"], "encoder_phase1.pt"))

    # ── Phase 2: Composite, freeze encoder ───────────────────────────────
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    ch2  = MildComposite(snr_range=(2, 20)).to(device)
    opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                     model.parameters()),
                              lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch2 = CosineAnnealingWarmRestarts(opt2, T_0=100)

    if verbose: print(f"\n  Phase 2: Composite (encoder FROZEN, train decoder+sync)")
    best_loss = _run_phase(model, ch2, cfg["phase2_epochs"], opt2, sch2,
                            cfg, "P2-sync", device, history, best_loss, verbose)

    # ── Phase 3: Composite, unfreeze all, low LR ────────────────────────
    for p in model.encoder.parameters():
        p.requires_grad_(True)

    ch3  = FullComposite(snr_range=cfg["snr_range"]).to(device)
    opt3 = torch.optim.AdamW(model.parameters(),
                              lr=cfg["lr_phase3"], weight_decay=cfg["weight_decay"])
    sch3 = CosineAnnealingWarmRestarts(opt3, T_0=100)

    if verbose: print(f"\n  Phase 3: Full composite (all params unfrozen, joint)")
    best_loss = _run_phase(model, ch3, cfg["phase3_epochs"], opt3, sch3,
                            cfg, "P3-joint", device, history, best_loss, verbose)

    # Save final
    torch.save(model.state_dict(),
               os.path.join(cfg["save_dir"], "final_model.pt"))
    with open(os.path.join(cfg["save_dir"], "history.json"), "w") as f:
        json.dump(history, f)

    # Load best checkpoint for return
    model.load_state_dict(
        torch.load(os.path.join(cfg["save_dir"], "best_model.pt"),
                   map_location=device))
    model.eval()

    if verbose:
        print(f"\n  Best loss: {best_loss:.4f}")
        print(f"  Saved → {cfg['save_dir']}/")

    return dict(model=model, history=history, n_params=n_params,
                flops=flops, cfg=cfg, best_loss=best_loss)


if __name__ == "__main__":
    train(verbose=True)
