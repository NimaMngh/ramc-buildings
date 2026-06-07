#!/usr/bin/env python3
"""
generate_perturbed_labels.py

Pre-generate RC-plant ground-truth labels for perturbed training inputs.

This is the offline data prep for the A1 (perturbation-only) ablation in
the Revision Plan. The A1 training loss includes

    L_pert = (1/BK) * sum_i sum_k || f_theta(z_i + delta_ik)
                                    - g_RC(z_i + delta_ik) ||^2_W

with g_RC being the RC plant (rc_ground_truth.py) evaluated on perturbed
inputs. Computing g_RC online in the training loop is infeasible (about
180 hours of wall clock for the full training set), so we pre-generate
K_label perturbed labels per training row once and load them at training
time via a wrapper dataset.

Usage:
    python generate_perturbed_labels.py \\
        --csv RAMC_training_data_N3.csv \\
        --k 4 \\
        --seed 12345 \\
        --out RAMC_perturbed_labels_K4.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rc_ground_truth import RCGroundTruthModel
from ramc_losses import sample_gaussian_perturbations


STATE_COLS   = ["T_air_k", "T_env_k", "T_int_k", "T_rad1_k", "T_rad2_k", "T_ret_k"]
CONTROL_COLS = ["T_supply_k", "mdot_k"]
TARGET_COLS  = ["T_air_k1", "T_env_k1", "T_int_k1", "T_rad1_k1", "T_rad2_k1", "T_ret_k1"]


def detect_solar_col(df: pd.DataFrame) -> str:
    for c in ("Q_solar_trans_k", "Q_solar_k", "I_solar_k"):
        if c in df.columns:
            return c
    raise KeyError("No solar column found in CSV")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Training data CSV")
    p.add_argument("--k", type=int, default=4, help="K_label perturbations per row")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", required=True, help="Output .npz path")
    # Match the training-time sigmas (defaults from main_experiment.LOSS_CONFIG)
    p.add_argument("--sigma-state",      type=float, default=1.0)
    p.add_argument("--sigma-rad-scale",  type=float, default=0.5)
    p.add_argument("--sigma-T-supply",   type=float, default=0.5)
    p.add_argument("--sigma-mdot",       type=float, default=0.01)
    p.add_argument("--sigma-T-out",      type=float, default=1.0)
    p.add_argument("--sigma-Q-solar",    type=float, default=500.0)
    p.add_argument("--sigma-Q-internal", type=float, default=200.0)
    args = p.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    solar = detect_solar_col(df)
    disturb_cols = ["T_out_k", solar, "Q_internal_k"]

    N = len(df)
    K = int(args.k)
    print(f"  rows={N}  K_label={K}  perturbed samples={N*K}  solar='{solar}'")

    states   = torch.from_numpy(df[STATE_COLS].values  ).float()
    controls = torch.from_numpy(df[CONTROL_COLS].values).float()
    disturb  = torch.from_numpy(df[disturb_cols].values).float()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Sampling Gaussian perturbations with training-time sigmas...")
    s_p, c_p, d_p = sample_gaussian_perturbations(
        states, controls, disturb,
        num_perturbations=K,
        sigma_state=args.sigma_state,
        sigma_rad_scale=args.sigma_rad_scale,
        sigma_T_supply=args.sigma_T_supply,
        sigma_mdot=args.sigma_mdot,
        sigma_T_out=args.sigma_T_out,
        sigma_Q_solar=args.sigma_Q_solar,
        sigma_Q_internal=args.sigma_Q_internal,
        clamp_physical=True,
        use_antithetic=True,
    )
    s_np = s_p.numpy().astype(np.float32)
    c_np = c_p.numpy().astype(np.float32)
    d_np = d_p.numpy().astype(np.float32)

    plant = RCGroundTruthModel(use_adaptive_substeps=True)

    print(f"Running RC plant on {N*K} perturbed inputs (this is the slow part)...")
    targets = np.zeros((N, K, len(TARGET_COLS)), dtype=np.float32)
    t0 = time.time()
    last_report = t0
    for i in range(N):
        for k in range(K):
            targets[i, k] = plant.step(s_np[i, k], c_np[i, k], d_np[i, k])
        if time.time() - last_report > 30.0:
            done = (i + 1) * K
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (N * K - done) / rate if rate > 0 else float("inf")
            print(f"  {done}/{N*K} ({100*done/(N*K):.1f}%)  "
                  f"rate={rate:.0f}/s  ETA={eta/60:.1f} min")
            last_report = time.time()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min ({(N*K)/elapsed:.0f} samples/s)")

    np.savez_compressed(
        args.out,
        perturbed_states=s_np,
        perturbed_controls=c_np,
        perturbed_disturb=d_np,
        perturbed_targets=targets,
        meta=np.array([{
            "seed": args.seed,
            "K_label": K,
            "N": N,
            "source_csv": args.csv,
            "sigmas": dict(
                state=args.sigma_state, rad_scale=args.sigma_rad_scale,
                T_supply=args.sigma_T_supply, mdot=args.sigma_mdot,
                T_out=args.sigma_T_out, Q_solar=args.sigma_Q_solar,
                Q_internal=args.sigma_Q_internal,
            ),
        }], dtype=object),
    )
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"Saved: {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
