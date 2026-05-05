"""
run_all.py — Master pipeline: train → evaluate → plot → report.

Usage:
  python run_all.py                    # full run
  python run_all.py --quick            # 150+200+100 epochs (fast test)
  python run_all.py --skip-train       # reload saved model
"""
import os, sys, json, argparse
import torch, numpy as np

from utils            import DEVICE, ensure_dirs, set_seed
from model            import AutoEncoder, estimate_flops, count_params
from train            import train, DEFAULT_CFG
from evaluation       import run_all_evaluations
from plots            import generate_all_plots
from report_generator import generate_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick",       action="store_true",
                   help="Short run: 150+200+100 epochs")
    p.add_argument("--skip-train",  action="store_true")
    p.add_argument("--skip-eval",   action="store_true")
    p.add_argument("--skip-plots",  action="store_true")
    p.add_argument("--n-symbols",   type=int, default=40000)
    p.add_argument("--save-dir",    default="results")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--M",           type=int, default=16)
    p.add_argument("--hidden",      type=int, default=256)
    return p.parse_args()


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def main():
    args = parse_args()
    ensure_dirs(args.save_dir, "plots")
    set_seed(args.seed)

    cfg = {**DEFAULT_CFG,
           "M": args.M, "hidden": args.hidden,
           "seed": args.seed, "save_dir": args.save_dir}

    if args.quick:
        cfg.update(phase1_epochs=150, phase2_epochs=200, phase3_epochs=100)

    # ── Train ────────────────────────────────────────────────────────────
    banner("Step 1 / 4 — Training")
    model_pt = os.path.join(args.save_dir, "best_model.pt")

    if args.skip_train and os.path.exists(model_pt):
        print(f"  Loading {model_pt}")
        model = AutoEncoder(cfg["M"], cfg["n_channel"], cfg["hidden"]).to(DEVICE)
        model.load_state_dict(torch.load(model_pt, map_location=DEVICE))
        model.eval()
        with open(os.path.join(args.save_dir, "history.json")) as f:
            history = json.load(f)
        n_params = count_params(model)
        flops    = estimate_flops(model)
    else:
        res      = train(cfg, verbose=True)
        model    = res["model"]
        history  = res["history"]
        n_params = res["n_params"]
        flops    = res["flops"]

    # ── Evaluate ─────────────────────────────────────────────────────────
    banner("Step 2 / 4 — BER Evaluation")
    ber_path = os.path.join(args.save_dir, "ber_results.json")
    if args.skip_eval and os.path.exists(ber_path):
        with open(ber_path) as f: results = json.load(f)
        print(f"  Loaded {ber_path}")
    else:
        results = run_all_evaluations(
            model, n_symbols=args.n_symbols,
            save_dir=args.save_dir, device=DEVICE)

    # ── Plots ────────────────────────────────────────────────────────────
    banner("Step 3 / 4 — Plots")
    if args.skip_plots:
        plot_paths = [os.path.join("plots", f)
                      for f in os.listdir("plots") if f.endswith(".png")]
        print(f"  Skipped (using {len(plot_paths)} existing plots)")
    else:
        plot_paths = generate_all_plots(model, results, history, DEVICE)

    # ── Report ───────────────────────────────────────────────────────────
    banner("Step 4 / 4 — Report")
    generate_report(model, results, history, cfg, n_params, flops,
                    plot_paths, save_dir=args.save_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    banner("DONE")
    snr  = np.array(results["snr_db"])
    idx  = int(np.searchsorted(snr, 10))
    print(f"  Device       : {DEVICE}")
    print(f"  Params       : {n_params:,}")
    print(f"  FLOPs        : {flops['total_flops']:,}")
    print(f"\n  BER @ SNR=10 dB:")
    for k in ["ae_awgn","ae_composite","ae_rayleigh","ae_impulsive",
              "bpsk_theory","qpsk_theory"]:
        arr = np.array(results.get(k, [1.0]))
        val = arr[idx] if idx < len(arr) else float("nan")
        print(f"    {k:30s}: {val:.4f}")
    print(f"\n  Files:")
    for f in ["best_model.pt","ber_results.json","report.txt","report.pdf"]:
        print(f"    {args.save_dir}/{f}")
    print(f"    plots/*.png  ({len(plot_paths)} figures)")


if __name__ == "__main__":
    main()
