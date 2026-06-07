"""
test_ablation_paths.py

Smoke tests for the new causal-ablation loss paths in ramc_losses.py.
Verifies that the diffs are correctly applied before any wall-clock
training run is started. Should finish in seconds on CPU.

Covers:
  - The new `mean` operator in risk_from_cost_samples.
  - The new loss_mode='pert_only' branch in calculate_ramc_loss,
    including the missing-labels error path and the manual-MSE check.
  - Regression: the default loss_mode='ramc' path with CVaR still works.
  - The early-exit at lambda_risk=0 still works for both modes.

Run from the directory containing ramc_losses.py:
    python test_ablation_paths.py
"""

import torch
import torch.nn as nn

from ramc_losses import calculate_ramc_loss, risk_from_cost_samples


class DummyModel(nn.Module):
    """Minimal stand-in for ThermalDynamicsNet. Matches the (s, c, d) -> y API."""
    def __init__(self, state_dim=6, control_dim=2, disturb_dim=3, output_dim=6):
        super().__init__()
        self.fc = nn.Linear(state_dim + control_dim + disturb_dim, output_dim)
        # Mirror ThermalDynamicsNet attributes used by the loss
        self.normalization_computed = False
        self.output_std = None

    def forward(self, s, c, d):
        return self.fc(torch.cat([s, c, d], dim=-1))


def _make_batch(B=8, sd=6, cd=2, dd=3, od=6, seed=0):
    torch.manual_seed(seed)
    s = torch.randn(B, sd) * 0.3 + 21.0
    c = torch.randn(B, cd) * 0.5 + torch.tensor([35.0, 0.5])
    d = torch.randn(B, dd) * 5.0
    y = s.clone()
    return s, c, d, y


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────

def test_mean_operator_dispatcher():
    """risk_from_cost_samples returns the arithmetic mean for 'mean'."""
    cost = torch.tensor([[1.0,  2.0,  3.0,  4.0],
                         [10.0, 20.0, 30.0, 40.0]])
    expected = torch.tensor([2.5, 25.0])
    for alias in ("mean", "expectation", "expected", "MEAN", "  Mean  "):
        out = risk_from_cost_samples(cost, risk_operator=alias)
        assert torch.allclose(out, expected), f"alias {alias!r} failed: {out}"
    # Sanity: differs from cvar at the same alpha
    cvar = risk_from_cost_samples(cost, risk_operator="cvar", cvar_alpha=0.9,
                                  cvar_method="empirical")
    assert not torch.allclose(out, cvar), "mean op should differ from CVaR"
    print("  PASS: mean operator dispatcher")


def test_mean_loss_path():
    """End-to-end calculate_ramc_loss with risk_operator='mean'."""
    model = DummyModel()
    s, c, d, y = _make_batch()
    out = calculate_ramc_loss(
        model, s, c, d, y,
        lambda_risk=0.001,
        risk_operator="mean",
        num_perturbations=4,
    )
    assert torch.isfinite(out["total_loss"])
    assert out["risk_loss"].numel() == 1
    assert out["risk_skipped"] is False
    assert out["loss_mode"] == "ramc"
    print(f"  PASS: mean loss path "
          f"(fid={out['fidelity_loss'].item():.4f}, "
          f"risk={out['risk_loss'].item():.4f}, "
          f"total={out['total_loss'].item():.4f})")


def test_pert_only_raises_without_labels():
    """loss_mode='pert_only' without perturbed_* args should raise."""
    model = DummyModel()
    s, c, d, y = _make_batch()
    try:
        calculate_ramc_loss(
            model, s, c, d, y,
            lambda_risk=0.5,
            loss_mode="pert_only",
        )
    except ValueError as e:
        assert "perturbed" in str(e).lower()
        print("  PASS: pert_only raises ValueError on missing labels")
        return
    raise AssertionError("Expected ValueError when perturbed labels missing")


def test_pert_only_matches_manual_mse():
    """pert_only risk_loss equals the manually-computed weighted MSE."""
    torch.manual_seed(42)
    B, K = 8, 4
    sd, cd, dd, od = 6, 2, 3, 6
    model = DummyModel(sd, cd, dd, od)
    s, c, d, y = _make_batch(B=B, sd=sd, cd=cd, dd=dd, od=od, seed=42)

    # Fake "RC-plant labels" — for the unit test these are just random tensors;
    # the math being verified is the loss aggregation, not their physical meaning.
    s_pert = s.unsqueeze(1) + torch.randn(B, K, sd) * 0.2
    c_pert = c.unsqueeze(1) + torch.randn(B, K, cd) * 0.1
    d_pert = d.unsqueeze(1) + torch.randn(B, K, dd) * 0.5
    y_pert = s_pert + torch.randn(B, K, od) * 0.05

    out = calculate_ramc_loss(
        model, s, c, d, y,
        lambda_risk=0.5,
        loss_mode="pert_only",
        perturbed_states=s_pert,
        perturbed_controls=c_pert,
        perturbed_disturb=d_pert,
        perturbed_targets=y_pert,
    )

    # Manual replication of the pert_only MSE
    with torch.no_grad():
        s_flat = s_pert.reshape(B * K, sd)
        c_flat = c_pert.reshape(B * K, cd)
        d_flat = d_pert.reshape(B * K, dd)
        y_flat = y_pert.reshape(B * K, od)
        # Match the eval-mode forward used inside calculate_ramc_loss
        was_training = model.training
        model.eval()
        preds = model(s_flat, c_flat, d_flat)
        if was_training:
            model.train()
        manual_mse = ((preds - y_flat) ** 2).mean()

    assert torch.allclose(out["risk_loss"], manual_mse, atol=1e-5), (
        f"risk_loss {out['risk_loss'].item()} != manual MSE {manual_mse.item()}"
    )
    assert out["risk_comfort_loss"].item() == 0.0
    assert out["risk_energy_loss"].item() == 0.0
    assert out["loss_mode"] == "pert_only"

    # Check total_loss = fidelity + lambda * risk
    expected_total = out["fidelity_loss"] + 0.5 * out["risk_loss"]
    assert torch.allclose(out["total_loss"], expected_total, atol=1e-6)

    print(f"  PASS: pert_only matches manual MSE "
          f"(risk_loss={out['risk_loss'].item():.6f})")


def test_default_cvar_path_unchanged():
    """Regression: default loss_mode='ramc' with CVaR still works."""
    model = DummyModel()
    s, c, d, y = _make_batch(B=8, seed=1)
    out = calculate_ramc_loss(
        model, s, c, d, y,
        lambda_risk=0.001,
        risk_operator="cvar",
        cvar_alpha=0.9,
        num_perturbations=16,
    )
    assert torch.isfinite(out["total_loss"])
    assert out["loss_mode"] == "ramc"
    assert out["risk_skipped"] is False
    print(f"  PASS: default CVaR path unchanged "
          f"(total={out['total_loss'].item():.4f})")


def test_lambda_zero_early_exit():
    """Early exit at lambda_risk=0 returns zero risk for both loss modes."""
    model = DummyModel()
    s, c, d, y = _make_batch(B=4)

    for lm in ("ramc", "pert_only"):
        out = calculate_ramc_loss(
            model, s, c, d, y,
            lambda_risk=0.0,
            loss_mode=lm,
            num_perturbations=4,
            skip_risk_if_lambda_zero=True,
        )
        assert out["risk_skipped"] is True
        assert out["risk_loss"].item() == 0.0
        assert torch.allclose(out["total_loss"], out["fidelity_loss"])
    print("  PASS: lambda=0 early exit works for both loss modes")


def main():
    print("Running ablation-path smoke tests...\n")
    test_mean_operator_dispatcher()
    test_mean_loss_path()
    test_pert_only_raises_without_labels()
    test_pert_only_matches_manual_mse()
    test_default_cvar_path_unchanged()
    test_lambda_zero_early_exit()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
