#!/usr/bin/env python3
"""
RAMC Model Loader for Phase 3 Closed-Loop Simulation
=====================================================

Loads RAMC-trained neural network checkpoints for use in MPC.

This loader is designed to work with the ThermalDynamicsNet architecture
used in RAMC Phase 2 training.

Robust checkpoint loading with weights_only fallback.

Author: Nima Monghasemi
Date: 2026-01-02 (Expert Review Fixes)
"""

import torch
from pathlib import Path
from typing import Optional, Union
import sys

# Add the neural network module path
NN_MODULE_PATH = Path(__file__).parent.parent.parent / "6_Neural Network Architecture"
if str(NN_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(NN_MODULE_PATH))

try:
    from thermal_dynamics_net import ThermalDynamicsNet
except ImportError:
    # Fallback: try relative import or raise helpful error
    raise ImportError(
        f"Could not import ThermalDynamicsNet. "
        f"Ensure thermal_dynamics_net.py is in: {NN_MODULE_PATH}"
    )


# =============================================================================
# RAMC Model Configuration (must match Phase 2 training)
# =============================================================================

RAMC_MODEL_CONFIG = {
    'state_dim': 6,
    'control_dim': 2,
    'disturbance_dim': 3,
    'hidden_dims': [256, 256, 128],
    'dropout_rate': 0.0,
    'use_residual': True,
}


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_ramc_model(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,  # CHANGED: Default to float32
    config_override: Optional[dict] = None,
    verbose: bool = True
) -> ThermalDynamicsNet:
    """
    Load a RAMC-trained model checkpoint.

    NOTE: Using float32 by default for ~3-10x faster Jacobian computation.
    The physical accuracy loss is negligible for MPC purposes.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Target device ('cpu' or 'cuda')
        dtype: Target dtype (torch.float32 or torch.float64)
        config_override: Optional dict to override default model config
        verbose: Print loading information
        
    Returns:
        ThermalDynamicsNet model ready for inference/linearization
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint format is incompatible
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if verbose:
        print(f"Loading RAMC model from: {checkpoint_path.name}")
    
    # Robust checkpoint loading with fallback
    # Phase 2 checkpoints may contain non-tensor data (history, epoch, etc.)
    # which can fail with weights_only=True depending on PyTorch version
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if verbose:
            print(f"  Loaded with weights_only=True")
    except Exception as e:
        if verbose:
            print(f"  weights_only=True failed ({type(e).__name__}), falling back...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if verbose:
                print(f"  Loaded with weights_only=False")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint with both weights_only modes. "
                f"First error: {e}, Second error: {e2}"
            )
    
    # Extract state dict (handle both formats)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Assume the dict itself is the state dict
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    # Build model config
    model_config = RAMC_MODEL_CONFIG.copy()
    if config_override:
        model_config.update(config_override)
    
    # Create model
    model = ThermalDynamicsNet(**model_config)
    
    # Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
        if verbose:
            print(f"  State dict loaded (strict=True)")
    except RuntimeError as e:
        if verbose:
            print(f"  Warning: Strict loading failed, trying non-strict: {e}")
        model.load_state_dict(state_dict, strict=False)
        if verbose:
            print(f"  State dict loaded (strict=False)")
    
    # Enable normalization if statistics are present in the checkpoint
    if hasattr(model, "enable_normalization_if_stats_present"):
        norm_enabled = model.enable_normalization_if_stats_present()
        if verbose and norm_enabled:
            print(f"  Normalization enabled from checkpoint statistics")
    
    # Set to evaluation mode and move to device/dtype
    model.eval()
    model.to(device=device, dtype=dtype)
    
    if verbose:
        print(f"  Model loaded successfully")
        print(f"  - States: {model.state_dim}, Controls: {model.control_dim}, Disturbances: {model.disturbance_dim}")
        print(f"  - Residual mode: {model.use_residual}")
        print(f"  - Device: {device}, Dtype: {dtype}")
    
    return model


def get_model_info(model: ThermalDynamicsNet) -> dict:
    """
    Get information about a loaded model.
    
    Args:
        model: ThermalDynamicsNet instance
        
    Returns:
        Dictionary with model information
    """
    info = {
        'state_dim': model.state_dim,
        'control_dim': model.control_dim,
        'disturbance_dim': model.disturbance_dim,
        'input_dim': model.input_dim,
        'use_residual': model.use_residual,
        'normalization_computed': model.normalization_computed,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # Add normalization stats if available
    if model.normalization_computed:
        info['input_mean_range'] = (
            float(model.input_mean.min()), 
            float(model.input_mean.max())
        )
        info['output_mean_range'] = (
            float(model.output_mean.min()), 
            float(model.output_mean.max())
        )
    
    return info


# =============================================================================
# Convenience function for loading multiple models
# =============================================================================

def load_model_set(
    checkpoint_dir: Union[str, Path],
    model_names: list,
    device: str = "cpu",
    verbose: bool = True
) -> dict:
    """
    Load multiple models from a checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_names: List of checkpoint filenames (without .pth extension)
        device: Target device
        verbose: Print loading information
        
    Returns:
        Dictionary mapping model names to loaded models
    """
    checkpoint_dir = Path(checkpoint_dir)
    models = {}
    
    for name in model_names:
        # Handle both with and without .pth extension
        if not name.endswith('.pth'):
            filename = f"{name}.pth"
        else:
            filename = name
            name = name.replace('.pth', '')
        
        checkpoint_path = checkpoint_dir / filename
        
        if checkpoint_path.exists():
            try:
                models[name] = load_ramc_model(
                    checkpoint_path, 
                    device=device, 
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"  Failed to load {name}: {e}")
        else:
            if verbose:
                print(f"  Checkpoint not found: {checkpoint_path}")
    
    return models


# =============================================================================
# Test function
# =============================================================================

def test_model_loading():
    """Test model loading with a sample checkpoint."""
    print("=" * 60)
    print("RAMC Model Loader Test")
    print("=" * 60)
    
    # Example checkpoint path (update as needed)
    RAMC_RUN_DIR = Path(
        r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
        r"\New Project for Risk Aware Model then Control\6_Neural Network Architecture"
        r"\results\RAMC_FULL_cvar_20260307_115447"
    )
    
    checkpoint_path = RAMC_RUN_DIR / "Fidelity_Baseline_rollout_a1.0_best.pth"
    
    if not checkpoint_path.exists():
        print(f"Test checkpoint not found: {checkpoint_path}")
        return None
    
    # Load model
    model = load_ramc_model(checkpoint_path, device='cpu', verbose=True)
    
    # Get info
    info = get_model_info(model)
    print(f"\nModel Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # Test forward pass
    import numpy as np
    x = torch.tensor([[21.0, 19.0, 20.5, 45.0, 45.0, 40.0]], dtype=torch.float64)
    u = torch.tensor([[50.0, 0.2]], dtype=torch.float64)
    d = torch.tensor([[-5.0, 5000.0, 1000.0]], dtype=torch.float64)
    
    with torch.no_grad():
        y = model(x, u, d)
    
    print(f"\nTest forward pass:")
    print(f"  Input x: {x.numpy().flatten()}")
    print(f"  Input u: {u.numpy().flatten()}")
    print(f"  Input d: {d.numpy().flatten()}")
    print(f"  Output y: {y.numpy().flatten()}")
    
    return model


if __name__ == "__main__":
    test_model_loading()
