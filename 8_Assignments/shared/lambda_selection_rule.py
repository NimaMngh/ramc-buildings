"""
λ-selection rule for RAMC hyperparameter choice.
==================================================

Implements the normalised utopia-point method described in Assignment 3.
Reads the Phase 2 decomposed evaluation CSV and selects λ*.

Column mapping (from decomposed_evaluation_full.csv):
  model_name, lambda, fidelity_loss_mean, risk_total_mean, risk_comfort_mean, ...

Used by: A3 (primary), A6 (Pareto analysis)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import csv


def load_decomposed_evaluation(csv_path: Path) -> List[Dict]:
    """Load the decomposed evaluation CSV from Phase 2."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_lambda_grid(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extract λ, fidelity loss, total risk, and comfort risk from rows.

    Returns: (lambdas, fidelity_losses, risk_totals, risk_comfort, model_names)
    Lambda = -1.0 for Raw MSE sentinel (excluded from utopia computation).
    """
    lambdas, fidelity, risk_total, risk_comfort, names = [], [], [], [], []

    for row in rows:
        names.append(row["model_name"])
        lambdas.append(float(row["lambda"]))
        fidelity.append(float(row["fidelity_loss_mean"]))
        risk_total.append(float(row["risk_total_mean"]))
        risk_comfort.append(float(row["risk_comfort_mean"]))

    return (np.array(lambdas), np.array(fidelity),
            np.array(risk_total), np.array(risk_comfort), names)


def utopia_point_select(
    lambdas: np.ndarray,
    fidelity_losses: np.ndarray,
    risk_scores: np.ndarray,
) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalised utopia-point method for λ selection.

    Both axes normalised to [0, 1] via min-max. Utopia = (0, 0).
    Returns (best_index, best_lambda, distances, fid_normalised, risk_normalised).
    """
    fid = np.asarray(fidelity_losses, dtype=float)
    risk = np.asarray(risk_scores, dtype=float)

    fid_range = max(fid.max() - fid.min(), 1e-12)
    risk_range = max(risk.max() - risk.min(), 1e-12)

    fid_norm = (fid - fid.min()) / fid_range
    risk_norm = (risk - risk.min()) / risk_range

    distances = np.sqrt(fid_norm**2 + risk_norm**2)
    best_idx = int(np.argmin(distances))

    return best_idx, float(lambdas[best_idx]), distances, fid_norm, risk_norm


def run_utopia_analysis(csv_path: Path, exclude_raw_mse: bool = True) -> Dict:
    """
    Full utopia-point λ-selection analysis.

    Parameters
    ----------
    csv_path : Path to decomposed_evaluation_full.csv
    exclude_raw_mse : if True, exclude Raw MSE (λ = -1) from normalisation

    Returns
    -------
    dict with selected lambdas, per-model results, and narrative text.
    """
    rows = load_decomposed_evaluation(csv_path)
    lambdas, fidelity, risk_total, risk_comfort, names = parse_lambda_grid(rows)

    # Filter: exclude Raw MSE (lambda = -1)
    mask = lambdas >= 0 if exclude_raw_mse else np.ones(len(lambdas), dtype=bool)
    f_lam = lambdas[mask]
    f_fid = fidelity[mask]
    f_rtot = risk_total[mask]
    f_rcom = risk_comfort[mask]
    f_names = [n for n, m in zip(names, mask) if m]

    # Primary: total risk
    best_idx_t, best_lam_t, dist_t, fid_norm_t, risk_norm_t = utopia_point_select(
        f_lam, f_fid, f_rtot
    )

    # Secondary: comfort risk only
    best_idx_c, best_lam_c, dist_c, fid_norm_c, risk_norm_c = utopia_point_select(
        f_lam, f_fid, f_rcom
    )

    # Build per-model results table
    model_results = []
    d_iter = iter(dist_t)
    for i, (lam, fid, rt, rc, name) in enumerate(
        zip(lambdas, fidelity, risk_total, risk_comfort, names)
    ):
        entry = {
            "model_name": name,
            "lambda": float(lam),
            "fidelity_loss": float(fid),
            "risk_total": float(rt),
            "risk_comfort": float(rc),
        }
        if mask[i]:
            d = float(next(d_iter))
            entry["utopia_distance"] = d
            entry["selected"] = (name == f_names[best_idx_t])
        else:
            entry["utopia_distance"] = None
            entry["selected"] = False
        model_results.append(entry)

    return {
        "selected_lambda_total_risk": best_lam_t,
        "selected_model_total_risk": f_names[best_idx_t],
        "utopia_distance_total_risk": float(dist_t[best_idx_t]),
        "selected_lambda_comfort_risk": best_lam_c,
        "selected_model_comfort_risk": f_names[best_idx_c],
        "utopia_distance_comfort_risk": float(dist_c[best_idx_c]),
        "model_results": model_results,
        # For plotting
        "filtered_lambdas": f_lam.tolist(),
        "filtered_names": f_names,
        "fid_normalised": fid_norm_t.tolist(),
        "risk_normalised": risk_norm_t.tolist(),
        "distances": dist_t.tolist(),
    }
