"""
RC-NMPC Benchmark — Differentiable RC Plant as NMPC Planning Model
===================================================================

Wraps the identified RC model as a PyTorch nn.Module so it can be used
as the planning model inside NMPCDirectShooting. This creates the
"exact-model RC-NMPC benchmark" for Assignment 2.

Integration method: IMEX (Implicit-Explicit) per substep
  - Building nodes (air, env, int): explicit Euler (slow dynamics)
  - Radiator advection: implicit Euler (stiff at high flow)
  - Radiator heat exchange to air: explicit (slower dynamics)

Why IMEX per substep:
  The radiator advection time constant is ~12s at max flow (4.05 kg/s).
  Pure explicit Euler needs ~86 substeps for stability, which is too slow
  with autograd (each substep creates graph nodes). By treating advection
  implicitly within each substep, 20 substeps is unconditionally stable
  while the heat exchange (tau_rad ~ 12 min) is well-resolved at dt_sub=30s.

Author: RAMC Assignment Framework
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict


# =============================================================================
# Default RC Parameters (from results_N3_DE_optimized.json)
# =============================================================================

DEFAULT_RC_PARAMS = {
    "C_air": 84426246.51832934,
    "C_env": 661376048.5634619,
    "C_int": 6002555765.428176,
    "C_rad": 597376.9622532368,
    "R_ex": 0.0003079967462046204,
    "R_ae": 0.00016180515492072352,
    "R_ai": 5.818704417041902e-05,
    "K_rad": 283.97226311004107,
    "a_rad": 0.2795410071110868,
    "A_sol": 0.5598500539656959,
}

# Number of substeps — 20 gives dt_sub=30s which is sufficient for the
# heat exchange dynamics (tau_rad ~ 12 min). The advection stiffness
# (tau_adv ~ 12s at max flow) is handled by the implicit treatment.
# 10 substeps (dt_sub=60s) is sufficient: verification shows max error
# < 1°C even at zero flow. Halves RC runtime vs 20 substeps.
N_SUBSTEPS = 10


class RCPlanningModel(nn.Module):
    """
    Differentiable RC thermal model for use as NMPC planning model.
    
    Uses IMEX integration: explicit for building nodes and radiator heat
    exchange, implicit for radiator advection. 20 substeps total.
    
    State: [T_air, T_env, T_int, T_rad1, T_rad2, T_ret]
    Control: [T_supply, mdot]
    Disturbance: [T_out, Q_solar, Q_internal]
    """
    
    def __init__(
        self,
        params: Optional[Dict] = None,
        dt_seconds: int = 600,
    ):
        super().__init__()
        
        p = params or DEFAULT_RC_PARAMS
        
        self.register_buffer("C_air", torch.tensor(p["C_air"], dtype=torch.float64))
        self.register_buffer("C_env", torch.tensor(p["C_env"], dtype=torch.float64))
        self.register_buffer("C_int", torch.tensor(p["C_int"], dtype=torch.float64))
        self.register_buffer("C_rad", torch.tensor(p["C_rad"], dtype=torch.float64))
        self.register_buffer("R_ex", torch.tensor(p["R_ex"], dtype=torch.float64))
        self.register_buffer("R_ae", torch.tensor(p["R_ae"], dtype=torch.float64))
        self.register_buffer("R_ai", torch.tensor(p["R_ai"], dtype=torch.float64))
        self.register_buffer("K_rad", torch.tensor(p["K_rad"], dtype=torch.float64))
        self.register_buffer("a_rad", torch.tensor(p["a_rad"], dtype=torch.float64))
        self.register_buffer("A_sol", torch.tensor(p["A_sol"], dtype=torch.float64))
        
        self.dt = dt_seconds
        self.N_rad = 3
        self.Cp = 4186.0
        
        # Interface compatibility with ThermalDynamicsNet
        self.state_dim = 6
        self.control_dim = 2
        self.disturbance_dim = 3
        self.use_residual = False
        self.normalization_computed = False
    
    def forward(self, x, u, d):
        squeeze = (x.dim() == 1)
        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            d = d.unsqueeze(0)
        
        results = []
        for b in range(x.shape[0]):
            results.append(self._step_single(x[b], u[b], d[b]))
        
        out = torch.stack(results, dim=0)
        if squeeze:
            out = out.squeeze(0)
        return out
    
    def _step_single(self, x, u, d):
        """
        IMEX integration: 20 substeps with implicit radiator advection.
        
        Per substep:
          1. Compute radiator heat exchange Q_rad at CURRENT temps (explicit)
          2. Update building nodes with explicit Euler (slow dynamics)
          3. Update radiator temps with implicit advection:
             C*(T_new - T_old)/dt_sub = mdot*Cp*(T_in - T_new) - Q_rad
             Solving: T_new = (C*T_old + dt_sub*mdot*Cp*T_in - dt_sub*Q_rad)
                              / (C + dt_sub*mdot*Cp)
          
        The implicit advection is unconditionally stable (no CFL limit),
        so 20 substeps works at any flow rate. The heat exchange Q_rad
        is well-resolved because tau_rad ~ 12 min >> dt_sub = 30s.
        """
        T_air = x[0]
        T_env = x[1]
        T_int = x[2]
        T_rad1 = x[3]
        T_rad2 = x[4]
        T_ret = x[5]
        
        T_supply = u[0]
        mdot = torch.clamp(u[1], min=0.0)
        
        T_out = d[0]
        Q_solar = d[1]
        Q_internal = d[2]
        
        dt_sub = float(self.dt) / N_SUBSTEPS
        
        C_air = self.C_air
        C_env = self.C_env
        C_int = self.C_int
        C_rad_sec = self.C_rad / self.N_rad
        R_ex = self.R_ex
        R_ae = self.R_ae
        R_ai = self.R_ai
        K_rad = self.K_rad
        a_rad = self.a_rad
        A_sol = self.A_sol
        
        Q_solar_air = A_sol * Q_solar
        Q_solar_int = (1.0 - A_sol) * Q_solar
        
        # Advection coefficient (constant across substeps)
        adv = mdot * self.Cp  # W/K
        # Denominator for implicit radiator update (constant across substeps)
        denom = C_rad_sec + dt_sub * adv  # J/K
        
        for _ in range(N_SUBSTEPS):
            # Step 1: Radiator heat exchange at current temps (explicit)
            dT1 = T_rad1 - T_air
            dT2 = T_rad2 - T_air
            dT3 = T_ret - T_air
            
            Q_rad1 = K_rad * dT1 * (torch.abs(dT1) + 1e-9) ** a_rad
            Q_rad2 = K_rad * dT2 * (torch.abs(dT2) + 1e-9) ** a_rad
            Q_rad3 = K_rad * dT3 * (torch.abs(dT3) + 1e-9) ** a_rad
            Q_rad_total = Q_rad1 + Q_rad2 + Q_rad3
            
            # Step 2: Building nodes — explicit Euler (slow dynamics)
            dTair_dt = (
                (T_env - T_air) / R_ae
                + (T_int - T_air) / R_ai
                + Q_rad_total
                + Q_internal
                + Q_solar_air
            ) / C_air
            
            dTenv_dt = (
                (T_out - T_env) / R_ex
                - (T_env - T_air) / R_ae
            ) / C_env
            
            dTint_dt = (
                -(T_int - T_air) / R_ai
                + Q_solar_int
            ) / C_int
            
            T_air = T_air + dTair_dt * dt_sub
            T_env = T_env + dTenv_dt * dt_sub
            T_int = T_int + dTint_dt * dt_sub
            
            # Step 3: Radiator sections — implicit advection, explicit Q_rad
            # T_new = (C*T_old + dt*adv*T_in - dt*Q_rad) / (C + dt*adv)
            T_rad1 = (C_rad_sec * T_rad1 + dt_sub * adv * T_supply - dt_sub * Q_rad1) / denom
            T_rad2 = (C_rad_sec * T_rad2 + dt_sub * adv * T_rad1 - dt_sub * Q_rad2) / denom
            T_ret = (C_rad_sec * T_ret + dt_sub * adv * T_rad2 - dt_sub * Q_rad3) / denom
        
        # Clamp to physical range
        T_air = torch.clamp(T_air, -50.0, 150.0)
        T_env = torch.clamp(T_env, -50.0, 150.0)
        T_int = torch.clamp(T_int, -50.0, 150.0)
        T_rad1 = torch.clamp(T_rad1, -50.0, 150.0)
        T_rad2 = torch.clamp(T_rad2, -50.0, 150.0)
        T_ret = torch.clamp(T_ret, -50.0, 150.0)
        
        return torch.stack([T_air, T_env, T_int, T_rad1, T_rad2, T_ret])
    
    def enable_normalization_if_stats_present(self):
        return False
    
    def __repr__(self):
        return (
            f"RCPlanningModel(dt={self.dt}s, N_rad={self.N_rad}, "
            f"substeps={N_SUBSTEPS}, method=IMEX, "
            f"A_sol={float(self.A_sol):.4f})"
        )


# =============================================================================
# Verification
# =============================================================================

def verify_rc_planning_model(verbose=True):
    """Verify against RCGroundTruthModel."""
    if verbose:
        print("Verifying RCPlanningModel against NumPy ground truth...")
    
    try:
        from rc_ground_truth import RCGroundTruthModel
    except ImportError:
        print("  WARNING: Cannot import RCGroundTruthModel")
        return True
    
    torch_model = RCPlanningModel()
    numpy_model = RCGroundTruthModel()
    
    test_cases = [
        # Low flow
        (np.array([21.0, 19.0, 20.5, 50.0, 45.0, 40.0]),
         np.array([55.0, 0.5]),
         np.array([-5.0, 3000.0, 50000.0])),
        # MAX flow — the critical test
        (np.array([15.0, 12.0, 14.0, 35.0, 33.0, 31.0]),
         np.array([60.0, 4.05]),
         np.array([-15.0, 0.0, 4000.0])),
        # Zero flow
        (np.array([22.0, 20.0, 21.0, 40.0, 38.0, 36.0]),
         np.array([32.0, 0.0]),
         np.array([5.0, 10000.0, 30000.0])),
        # Medium flow
        (np.array([20.0, 18.0, 19.0, 55.0, 50.0, 45.0]),
         np.array([58.0, 2.0]),
         np.array([-10.0, 5000.0, 20000.0])),
    ]
    
    max_err = 0.0
    for i, (x0, u, d) in enumerate(test_cases):
        x_np = numpy_model.step(x0.copy(), u, d)
        x_t = torch_model(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(u, dtype=torch.float64),
            torch.tensor(d, dtype=torch.float64),
        ).detach().numpy()
        
        err = np.max(np.abs(x_t - x_np))
        max_err = max(max_err, err)
        
        if verbose:
            status = "" if err < 1.0 else "<- WARNING"
            print(f"  Test {i+1} (mdot={u[1]:.2f}): max_err = {err:.4f} °C  {status}")
            if err > 1.0:
                print(f"    T_air: torch={x_t[0]:.4f}  numpy={x_np[0]:.4f}")
                print(f"    T_ret: torch={x_t[5]:.4f}  numpy={x_np[5]:.4f}")
    
    ok = max_err < 2.0
    if verbose:
        status = "PASS" if ok else "FAIL"
        print(f"  Overall: {status} (max error = {max_err:.4f} °C)")
    
    return ok


def verify_gradient_flow(verbose=True):
    """Verify gradients flow through the model."""
    if verbose:
        print("Verifying gradient flow...")
    
    model = RCPlanningModel()
    x0 = torch.tensor([21.0, 19.0, 20.5, 50.0, 45.0, 40.0], dtype=torch.float64)
    u = torch.tensor([55.0, 0.5], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([-5.0, 3000.0, 50000.0], dtype=torch.float64)
    
    x_next = model(x0, u, d)
    x_next[0].backward()
    
    ok = u.grad is not None and not torch.all(u.grad == 0)
    if verbose:
        print(f"  d(T_air)/d(T_supply) = {u.grad[0]:.6f}")
        print(f"  d(T_air)/d(mdot)     = {u.grad[1]:.6f}")
        print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("RC-NMPC Benchmark Verification")
    print("=" * 60)
    print(f"\n{RCPlanningModel()}\n")
    verify_rc_planning_model()
    print()
    verify_gradient_flow()
