# -*- coding: utf-8 -*-
"""
Thermal Dynamics Neural Network for RAMC.

create_dataloaders returns split indices for test-only evaluation, and
PerturbedLabelDataset wraps the dataset for A1 ablation support.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple, Dict

# =============================================================================
# Neural Network Class
# =============================================================================

class ThermalDynamicsNet(nn.Module):
    """
    Thermal dynamics NN with residual learning.
    """
    
    def __init__(self, 
                 state_dim: int = 6,
                 control_dim: int = 2,
                 disturbance_dim: int = 3,
                 hidden_dims: list = [128, 128, 64],
                 use_residual: bool = True,
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.use_residual = bool(use_residual)
        
        self.n_states = state_dim
        self.n_controls = control_dim
        self.n_disturbances = disturbance_dim
        self.input_dim = state_dim + control_dim + disturbance_dim
        
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, state_dim))
        self.network = nn.Sequential(*layers)
        
        # Normalization buffers
        self.register_buffer('input_mean', torch.zeros(self.input_dim))
        self.register_buffer('input_std', torch.ones(self.input_dim))
        self.register_buffer('output_mean', torch.zeros(state_dim))
        self.register_buffer('output_std', torch.ones(state_dim))
        
        self.normalization_computed = False
        self._residual_mode_locked = False
    
    def forward(self, states, controls, disturbances):
        inputs = torch.cat([states, controls, disturbances], dim=1)
        
        if self.normalization_computed:
            x_norm = (inputs - self.input_mean) / (self.input_std + 1e-8)
            output_norm = self.network(x_norm)
            output_denorm = output_norm * self.output_std + self.output_mean
        else:
            output_denorm = self.network(inputs)
        
        if self.use_residual:
            return states + output_denorm
        else:
            return output_denorm
    
    def forward_phys(self, x, u, d):
        """Single-sample prediction for linearization/MPC."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        if d.dim() == 1:
            d = d.unsqueeze(0)
        y = self.forward(x, u, d)
        return y.squeeze(0)
    
    def compute_normalization(self, dataloader):
        """Compute statistics for residual or absolute targets."""
        print(f"Computing normalization (residual_mode={self.use_residual})...")
        
        all_inputs = []
        all_targets = []
        
        device = next(self.parameters()).device
        self.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                states, controls, disturbances, targets = batch[:4]
                states = states.to(device)
                controls = controls.to(device)
                disturbances = disturbances.to(device)
                targets = targets.to(device)
                
                inputs = torch.cat([states, controls, disturbances], dim=1)
                all_inputs.append(inputs)
                
                if self.use_residual:
                    target_data = targets - states
                else:
                    target_data = targets
                    
                all_targets.append(target_data)
        
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        self.input_mean.data = all_inputs.mean(dim=0)
        self.input_std.data = all_inputs.std(dim=0)
        self.output_mean.data = all_targets.mean(dim=0)
        self.output_std.data = all_targets.std(dim=0)
        
        self.input_std.data = torch.clamp(self.input_std.data, min=1e-8)
        self.output_std.data = torch.clamp(self.output_std.data, min=1e-8)
        
        self.normalization_computed = True
        self._residual_mode_locked = True
        
        print(f"Normalization computed (residual={self.use_residual}):")
        print(f"  Input  mean range: [{self.input_mean.min():.2f}, {self.input_mean.max():.2f}]")
        print(f"  Output mean range: [{self.output_mean.min():.2f}, {self.output_mean.max():.2f}]")

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'output_dim': self.state_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'normalization_computed': self.normalization_computed,
            'use_residual': self.use_residual,
            'residual_mode_locked': self._residual_mode_locked
        }

    def enable_normalization_if_stats_present(self):
        """
        Heuristic to enable normalization after loading checkpoint.
        
        The normalization_computed flag is a Python bool (not a buffer) so it's
        not saved/restored with state_dict. This method detects if normalization
        statistics were loaded and enables normalization accordingly.
        
        Checks both mean ≠ 0 and std ≠ 1 to handle edge cases where std might
        be close to 1 by coincidence.
        
        Returns:
            bool: True if normalization was enabled, False otherwise
        """
        # Check if input/output mean differ from default (zeros)
        input_mean_differs = torch.any(self.input_mean.abs() > 1e-6).item()
        output_mean_differs = torch.any(self.output_mean.abs() > 1e-6).item()
        
        # Check if input/output std differ from default (ones)
        input_std_differs = torch.any((self.input_std - 1.0).abs() > 1e-6).item()
        output_std_differs = torch.any((self.output_std - 1.0).abs() > 1e-6).item()
        
        mean_differs = input_mean_differs or output_mean_differs
        std_differs = input_std_differs or output_std_differs
        
        if mean_differs or std_differs:
            self.normalization_computed = True
            self._residual_mode_locked = True
            return True
        return False


# =============================================================================
# Dataset Class - P7: Episode Segmentation
# =============================================================================

class RCModelDataset(Dataset):
    """
    P7: Added episode segmentation based on timestamp discontinuities.
    P7: valid_rollout_starts() helper for multi-step rollout evaluation.
    P8: Correct column name mappings for clamp bounds building.
    """
    
    # P8: Column name mappings for external use (e.g., clamp bounds)
    STATE_COLS = ['T_air_k', 'T_env_k', 'T_int_k', 'T_rad1_k', 'T_rad2_k', 'T_ret_k']
    CONTROL_COLS = ['T_supply_k', 'mdot_k']
    DISTURBANCE_COLS = ['T_out_k', 'Q_solar_trans_k', 'Q_internal_k']
    TARGET_COLS = ['T_air_k1', 'T_env_k1', 'T_int_k1', 'T_rad1_k1', 'T_rad2_k1', 'T_ret_k1']
    
    def __init__(self, csv_file_path: str, expected_dt_seconds: int = 600):
        """
        P7: Added expected_dt_seconds for episode segmentation.
        """
        print(f"Loading data from: {csv_file_path}")
        
        self.df = pd.read_csv(csv_file_path)
        self.expected_dt_seconds = int(expected_dt_seconds)
        
        # P7: Parse timestamp and compute episode IDs
        if "Timestamp" in self.df.columns:
            self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"])
            # Do NOT sort by timestamp — episodes are randomly ordered
            # and sorting would interleave samples from different episodes
            print("  Timestamp column detected - preserving original row order")
            self.timestamps = self.df["Timestamp"].to_numpy()
        else:
            self.timestamps = None
        
        # Episode segmentation: use existing episode_id if available, else infer from dt
        if "episode_id" in self.df.columns:
            self.episode_id = self.df["episode_id"].to_numpy().astype(int)
            n_episodes = int(self.df["episode_id"].nunique())
            print(f"  Using existing episode_id column: {n_episodes} episodes")
        else:
            if self.timestamps is not None:
                dt = self.df["Timestamp"].diff().dt.total_seconds()
                dt_tolerance = float(self.expected_dt_seconds) * 0.1
                new_episode = dt.isna() | ((dt - float(self.expected_dt_seconds)).abs() > dt_tolerance)
                self.df["episode_id"] = new_episode.cumsum().astype(int)
                self.episode_id = self.df["episode_id"].to_numpy()
                n_episodes = int(self.df["episode_id"].nunique())
                print(f"  Episode segmentation from timestamps: {n_episodes} episodes detected")
            else:
                self.episode_id = None
                print("  No episode segmentation available")
        
        self.state_cols = self.STATE_COLS
        self.control_cols = self.CONTROL_COLS
        self.disturbance_cols = self.DISTURBANCE_COLS
        self.target_cols = self.TARGET_COLS
        
        required_cols = self.state_cols + self.control_cols + self.disturbance_cols + self.target_cols
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        self.states = torch.tensor(self.df[self.state_cols].values, dtype=torch.float32)
        self.controls = torch.tensor(self.df[self.control_cols].values, dtype=torch.float32)
        self.disturbances = torch.tensor(self.df[self.disturbance_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(self.df[self.target_cols].values, dtype=torch.float32)
        
        self.has_bounds = ('Tmin' in self.df.columns) and ('Tmax' in self.df.columns)
        if self.has_bounds:
            print("  Found 'Tmin' and 'Tmax' columns")
            self.Tmin = torch.tensor(self.df['Tmin'].values, dtype=torch.float32)
            self.Tmax = torch.tensor(self.df['Tmax'].values, dtype=torch.float32)
        else:
            print("  No Tmin/Tmax - will use fixed bounds")
            self.Tmin = None
            self.Tmax = None
            
        print(f"Dataset loaded:")
        print(f"  Samples: {len(self.df)}")
        print(f"  States: {self.states.shape}")
        print(f"  Controls: {self.controls.shape}")
        print(f"  Disturbances: {self.disturbances.shape}")
        print(f"  Targets: {self.targets.shape}")
        
        self._print_data_summary()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.has_bounds:
            return (
                self.states[idx],
                self.controls[idx],
                self.disturbances[idx],
                self.targets[idx],
                self.Tmin[idx],
                self.Tmax[idx]
            )
        else:
            return (
                self.states[idx],
                self.controls[idx], 
                self.disturbances[idx],
                self.targets[idx]
            )
    
    def valid_rollout_starts(self, horizon: int) -> np.ndarray:
        """
        P7: Return indices i such that i..i+horizon-1 stays in same episode.
        
        Args:
            horizon: Number of steps for the rollout
            
        Returns:
            Array of valid starting indices
        """
        H = int(horizon)
        
        if self.episode_id is None:
            # No episode info - assume all contiguous
            return np.arange(0, len(self.df) - H, dtype=int)
        
        ep = self.episode_id
        valid = []
        for i in range(0, len(ep) - H):
            if ep[i] == ep[i + H - 1]:
                valid.append(i)
        return np.asarray(valid, dtype=int)
    
    def get_episode_lengths(self) -> Dict[int, int]:
        """P7: Get length of each episode."""
        if self.episode_id is None:
            return {0: len(self.df)}
        
        unique, counts = np.unique(self.episode_id, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def _print_data_summary(self):
        print(f"\nData Summary:")
        
        for i, col in enumerate(self.state_cols):
            values = self.states[:, i]
            print(f"  {col}: [{values.min():.1f}, {values.max():.1f}] C")
        
        for i, col in enumerate(self.control_cols):
            values = self.controls[:, i]
            if 'T_supply' in col:
                print(f"  {col}: [{values.min():.1f}, {values.max():.1f}] C")
            else:
                print(f"  {col}: [{values.min():.3f}, {values.max():.3f}] kg/s")
        
        for i, col in enumerate(self.disturbance_cols):
            values = self.disturbances[:, i]
            if 'T_out' in col:
                print(f"  {col}: [{values.min():.1f}, {values.max():.1f}] C")
            else:
                print(f"  {col}: [{values.min():.0f}, {values.max():.0f}] W")


class PerturbedLabelDataset(Dataset):
    """
    Wraps an RCModelDataset and appends pre-generated perturbed inputs and
    RC-plant ground-truth labels to each sample, for the A1 ablation in
    the Revision Plan.

    The .npz file must have been produced by generate_perturbed_labels.py
    with row ordering matching the base dataset (which preserves the CSV
    row order). The wrapper checks N at construction time.

    Each sample is returned as the base tuple (4 or 6 tensors) followed
    by four extra tensors:
        perturbed_states     [K_label, state_dim]
        perturbed_controls   [K_label, control_dim]
        perturbed_disturb    [K_label, disturbance_dim]
        perturbed_targets    [K_label, output_dim]
    """

    def __init__(self, base_dataset, perturbed_labels_path):
        self.base = base_dataset
        data = np.load(perturbed_labels_path, allow_pickle=True)
        self.s_pert = torch.from_numpy(data["perturbed_states"]).float()
        self.c_pert = torch.from_numpy(data["perturbed_controls"]).float()
        self.d_pert = torch.from_numpy(data["perturbed_disturb"]).float()
        self.y_pert = torch.from_numpy(data["perturbed_targets"]).float()

        N_base = len(self.base)
        N_pert = self.s_pert.shape[0]
        if N_base != N_pert:
            raise ValueError(
                f"Perturbed-labels file has {N_pert} rows but base dataset "
                f"has {N_base}. Regenerate the .npz against the current CSV."
            )
        self.K_label = int(self.s_pert.shape[1])
        print(f"PerturbedLabelDataset: N={N_base}, K_label={self.K_label}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        base_item = self.base[idx]
        if not isinstance(base_item, (tuple, list)):
            base_item = (base_item,)
        return tuple(base_item) + (
            self.s_pert[idx],
            self.c_pert[idx],
            self.d_pert[idx],
            self.y_pert[idx],
        )
        
    def __getattr__(self, name):
        """Pass through attributes like df, valid_rollout_starts, etc."""
        return getattr(self.base, name)


# =============================================================================
# Rollout Sequence Sampler (for rollout-aware RAMC training)
# =============================================================================

class RolloutSequenceSampler:
    """
    Samples batches of H_r-step consecutive sequences from valid rollout start
    positions within the training set.

    Used alongside the standard DataLoader (not replacing it).  Each call to
    ``sample_batch`` returns one batch of B_r independent rollout sequences,
    each of length H_r.

    Args:
        dataset:          RCModelDataset instance (full dataset)
        train_indices:    1-D integer array of indices belonging to the training
                          split.  Only starts fully contained within this split
                          are eligible.
        rollout_horizon:  H_r — number of steps per rollout (default 6 = 1 hour
                          at 10-minute timesteps)
        batch_size:       B_r — sequences per sampled batch (default 64)
        seed:             Random seed for reproducibility
    """

    def __init__(
        self,
        dataset: "RCModelDataset",
        train_indices: np.ndarray,
        rollout_horizon: int = 6,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.rollout_horizon = int(rollout_horizon)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

        # Build a fast lookup set for training indices
        train_index_set = set(train_indices.tolist())

        # Get all episode-valid starts from the full dataset
        all_valid = dataset.valid_rollout_starts(self.rollout_horizon)

        # Keep only starts where start AND all H_r subsequent steps are in train
        self.valid_starts: np.ndarray = np.array(
            [
                s for s in all_valid
                if all((s + h) in train_index_set for h in range(self.rollout_horizon + 1))
            ],
            dtype=np.int64,
        )

        if len(self.valid_starts) == 0:
            raise RuntimeError(
                f"RolloutSequenceSampler: no valid rollout starts found "
                f"(train_indices size={len(train_indices)}, H_r={self.rollout_horizon}). "
                f"Check episode segmentation and that train_indices covers enough "
                f"consecutive samples."
            )

        print(
            f"[RolloutSequenceSampler] {len(self.valid_starts)} valid starts "
            f"(H_r={self.rollout_horizon}, B_r={self.batch_size})"
        )

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one batch of B_r rollout sequences.

        Returns:
            x0:               [B_r, nx]        Initial states (ground truth)
            controls_seq:     [B_r, H_r, nu]   Ground-truth control sequences
            disturbances_seq: [B_r, H_r, nd]   Ground-truth disturbance seqs
            targets_seq:      [B_r, H_r, nx]   Ground-truth next-state seqs

        Note: targets_seq[:, h] is the ground truth at step h+1, i.e.
        dataset.targets[start + h], which equals dataset.states[start + h + 1]
        by dataset convention.
        """
        B_r = min(self.batch_size, len(self.valid_starts))
        chosen = self.rng.choice(self.valid_starts, size=B_r, replace=(B_r > len(self.valid_starts)))

        ds = self.dataset
        H_r = self.rollout_horizon

        x0_list    : List[torch.Tensor] = []
        ctrl_list  : List[torch.Tensor] = []
        dist_list  : List[torch.Tensor] = []
        tgt_list   : List[torch.Tensor] = []

        for s in chosen:
            x0_list.append(ds.states[s])                                          # [nx]
            ctrl_list.append(ds.controls[s : s + H_r])                            # [H_r, nu]
            dist_list.append(ds.disturbances[s : s + H_r])                        # [H_r, nd]
            tgt_list.append(ds.targets[s : s + H_r])                              # [H_r, nx]

        x0             = torch.stack(x0_list,   dim=0)   # [B_r, nx]
        controls_seq   = torch.stack(ctrl_list,  dim=0)  # [B_r, H_r, nu]
        disturbances_seq = torch.stack(dist_list, dim=0) # [B_r, H_r, nd]
        targets_seq    = torch.stack(tgt_list,  dim=0)   # [B_r, H_r, nx]

        return x0, controls_seq, disturbances_seq, targets_seq


# =============================================================================
# P8: Clamp Bounds Builder (correct column names)
# =============================================================================

def build_clamp_bounds_from_dataset(
    dataset: RCModelDataset, 
    qlo: float = 0.001, 
    qhi: float = 0.999,
    indices: Optional[np.ndarray] = None,  # Optional subset for train-only bounds
) -> Dict:
    """
    P8: Build clamp bounds from dataset quantiles using correct column names.
    
    Added optional indices parameter to compute bounds from a subset
    (e.g., training data only) rather than the full dataset.
    
    Args:
        dataset: RCModelDataset instance
        qlo: Lower quantile (default 0.1%)
        qhi: Upper quantile (default 99.9%)
        indices: Optional array of indices to use (e.g., train_indices)
        
    Returns:
        Dict with clamp bounds for state, control, dist
    """
    if indices is not None:
        df = dataset.df.iloc[indices]
    else:
        df = dataset.df
    
    def qbounds(series):
        return float(series.quantile(qlo)), float(series.quantile(qhi))
    
    return {
        "state": {
            "T_air":  qbounds(df["T_air_k"]),
            "T_env":  qbounds(df["T_env_k"]),
            "T_int":  qbounds(df["T_int_k"]),
            "T_rad1": qbounds(df["T_rad1_k"]),
            "T_rad2": qbounds(df["T_rad2_k"]),
            "T_ret":  qbounds(df["T_ret_k"]),
        },
        "control": {
            "T_supply": qbounds(df["T_supply_k"]),
            "mdot":     qbounds(df["mdot_k"]),
        },
        "dist": {
            "T_out":      qbounds(df["T_out_k"]),
            "Q_solar":    qbounds(df["Q_solar_trans_k"]),
            "Q_internal": qbounds(df["Q_internal_k"]),
        }
    }


# =============================================================================
# DataLoader Function - Now returns split indices
# =============================================================================

def create_dataloaders(
    csv_file_path: str, 
    batch_size: int = 512,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_state: int = 42,
    device: str = 'cpu',
    split_mode: str = 'time',
    expected_dt_seconds: int = 600,
) -> Tuple[DataLoader, DataLoader, DataLoader, RCModelDataset, Dict[str, np.ndarray]]:
    """
    Create dataloaders with time-based or random splitting.
    
    Now returns split indices for test-only rollout evaluation and
    train-only clamp bounds computation.
    
    P7: Passes expected_dt_seconds to dataset.
    
    Returns:
        train_loader, val_loader, test_loader, full_dataset, split_indices
        
        split_indices is a dict with keys: 'train', 'val', 'test'
    """
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    
    full_dataset = RCModelDataset(csv_file_path, expected_dt_seconds=expected_dt_seconds)
    n_samples = len(full_dataset)
    
    if split_mode == 'time':
        print(f"\nUsing TIME-BASED PARTITION (sequential)")
        n_train = int(train_split * n_samples)
        n_val = int(val_split * n_samples)
        
        train_indices = np.arange(0, n_train)
        val_indices = np.arange(n_train, n_train + n_val)
        test_indices = np.arange(n_train + n_val, n_samples)
        
    elif split_mode == 'random':
        print(f"\nUsing RANDOM PARTITION")
        indices = np.arange(n_samples)
        
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_split, random_state=random_state, shuffle=True
        )
        
        val_ratio = val_split / (train_split + val_split)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio, random_state=random_state, shuffle=True
        )
        
        # Convert to numpy arrays and sort for consistency
        train_indices = np.sort(np.array(train_indices))
        val_indices = np.sort(np.array(val_indices))
        test_indices = np.sort(np.array(test_indices))
    else:
        raise ValueError(f"split_mode must be 'time' or 'random'")
    
    print(f"Data Split:")
    print(f"  Training:   {len(train_indices)} ({len(train_indices)/n_samples*100:.1f}%)")
    print(f"  Validation: {len(val_indices)} ({len(val_indices)/n_samples*100:.1f}%)")
    print(f"  Test:       {len(test_indices)} ({len(test_indices)/n_samples*100:.1f}%)")
    
    # Store split indices for later use
    split_indices = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    pin_memory = (device == 'cuda')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader, full_dataset, split_indices


if __name__ == '__main__':
    print("ThermalDynamicsNet module - use test_data_and_model() for testing")