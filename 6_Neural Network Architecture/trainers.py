# -*- coding: utf-8 -*-
"""
RAMC Training Framework.

The trainer sanitizes the risk_operator string in trainer naming and skips
the expensive risk computation when lambda=0.
"""

from __future__ import annotations

import os
import time
import json
import random
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from thermal_dynamics_net import ThermalDynamicsNet, RolloutSequenceSampler

from ramc_losses import (
    calculate_ramc_loss,
    evaluate_model_predictions,
    evaluate_on_loader,
    compute_fidelity_loss,
    compute_raw_mse,
    compute_rollout_fidelity_loss,
)


# ---------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------

class BaseTrainer:
    """Base class for all trainers with common checkpointing and plotting."""

    def __init__(
        self,
        model: ThermalDynamicsNet,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        save_dir: str = "models",
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)

        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=15,
            factor=0.5,
        )

        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "learning_rate": [],
            "train_loss": [],
            "val_loss": [],
        }

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def save_model(self, filepath: str, epoch: int, val_metric: float):
        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metric": float(val_metric),
            "history": self.history,
            "model_config": (
                self.model.get_model_info() if hasattr(self.model, "get_model_info") else {}
            ),
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath: str, strict: bool = True):
        checkpoint = None
        state_dict = None
    
        try:
            loaded_data = torch.load(filepath, map_location=self.device, weights_only=True)
            if isinstance(loaded_data, dict):
                state_dict = loaded_data.get("model_state_dict", loaded_data)
                checkpoint = loaded_data
            else:
                state_dict = loaded_data
                checkpoint = {}
        except Exception:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            if not isinstance(checkpoint, dict):
                raise RuntimeError(f"Unrecognized checkpoint format: {filepath}")
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
            if state_dict is None:
                raise RuntimeError(f"Unrecognized checkpoint format: {filepath}")
    
        self.model.load_state_dict(state_dict, strict=strict)
        
        # Enable normalization if statistics were loaded
        if hasattr(self.model, 'enable_normalization_if_stats_present'):
            self.model.enable_normalization_if_stats_present()
    
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass
    
        if isinstance(checkpoint, dict) and "history" in checkpoint:
            self.history = checkpoint["history"]
    
        return checkpoint



    def plot_training_history(self, save_path: Optional[str] = None):
        if len(self.history.get("train_loss", [])) == 0:
            print("No training history to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = self.history["epoch"]
        ax1.plot(epochs, self.history["train_loss"], "b-", label="Train Loss", alpha=0.85)
        ax1.plot(epochs, self.history["val_loss"], "r-", label="Val Loss", alpha=0.85)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{self.__class__.__name__} - Training Progress")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.history["learning_rate"], "g-", alpha=0.85)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------
# RAMC trainer - Sanitize risk_operator + skip risk when lambda=0
# ---------------------------------------------------------------------

class RAMCTrainer(BaseTrainer):
    """RAMC trainer with P1-P10 improvements, performance fixes, and rollout-aware fidelity."""

    def __init__(
        self,
        model: ThermalDynamicsNet,
        lambda_risk: float = 1.0,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        save_dir: str = "models",
        grad_clip_norm: Optional[float] = 1.0,
        weight_decay: float = 1e-5,
        **loss_kwargs,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            device=device,
            save_dir=save_dir,
            weight_decay=weight_decay,
        )

        self.target_lambda = float(lambda_risk)
        self.current_lambda = 0.0
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None
        self.loss_kwargs: Dict[str, Any] = dict(loss_kwargs)

        # Option A fix: when True, dataset Tmin/Tmax columns are ignored during
        # training and validation so that the CVaR stage cost always uses the
        # fixed comfort_bounds from loss_kwargs (default (20, 22) °C).  This
        # decouples the training objective from whatever bounds were baked into
        # the dataset at generation time (which used OCC_TARGET=22, DEADBAND=0.5
        # -> [21.5, 22.5] °C, mismatched with Phase 3 evaluation).
        self.ignore_dataset_bounds: bool = bool(
            self.loss_kwargs.pop("ignore_dataset_bounds", False)
        )

        # Rollout-aware fidelity parameters (Section 5.4 of design spec).
        # Popped from loss_kwargs so they don't leak into calculate_ramc_loss.
        self.alpha_rollout: float = float(
            self.loss_kwargs.pop("alpha_rollout", 0.0)
        )
        self.rollout_horizon: int = int(
            self.loss_kwargs.pop("rollout_horizon", 6)
        )
        self.rollout_batch_size: int = int(
            self.loss_kwargs.pop("rollout_batch_size", 64)
        )
        self.rollout_step_weights: str = str(
            self.loss_kwargs.pop("rollout_step_weights", "linear")
        )

        # Sanitize risk_operator string
        op = str(self.loss_kwargs.get("risk_operator", "std")).strip().lower()
        loss_mode = str(self.loss_kwargs.get("loss_mode", "ramc")).strip().lower()

        # P1: Better naming — include rollout suffix when active
        if self.target_lambda == 0.0:
            base_name = "Fidelity_Baseline"
        elif loss_mode == "pert_only":
            base_name = f"PertOnly_gamma_{self.target_lambda}"
        elif op in ("mean", "expectation", "expected"):
            base_name = f"MeanCost_lambda_{self.target_lambda}"
        else:
            base_name = f"RAMC_lambda_{self.target_lambda}_op_{op}"

        if self.alpha_rollout > 0.0:
            self.trainer_name = f"{base_name}_rollout_a{self.alpha_rollout}"
        else:
            self.trainer_name = base_name

        self.best_cost_val = float("inf")
        self.best_cost_epoch = 0
        self.best_risk_val = float("inf")
        self.best_risk_epoch = 0

        # P1/P10: Extended history tracking
        self.history.update({
            "current_lambda": [],
            "train_total_loss": [],
            "train_fidelity_loss": [],
            "train_mse_raw": [],
            "train_risk_loss": [],
            "train_expected_cost": [],
            "train_rollout_loss": [],      # Rollout-aware fidelity term
            "val_total_loss": [],
            "val_total_loss_target": [],
            "val_fidelity_loss": [],
            "val_mse_raw": [],
            "val_risk_loss": [],
            "val_risk_comfort_loss": [],
            "val_risk_energy_loss": [],
            "val_expected_cost": [],
            "val_expected_comfort": [],
            "val_expected_energy": [],
            "val_t_air_rmse": [],
        })

        print(f"RAMC Trainer initialized: {self.trainer_name}")

    @staticmethod
    def _unpack_batch(batch):
        """
        Accept the four legal batch shapes:
            4-tuple  : (s, c, d, y)
            6-tuple  : (s, c, d, y, Tmin, Tmax)
            8-tuple  : (s, c, d, y, s_pert, c_pert, d_pert, y_pert)
            10-tuple : (s, c, d, y, Tmin, Tmax, s_pert, c_pert, d_pert, y_pert)
        Returns a flat tuple
            (s, c, d, y, Tmin, Tmax, s_pert, c_pert, d_pert, y_pert)
        with None for any field that was absent.
        """
        if not isinstance(batch, (tuple, list)):
            raise ValueError(f"Expected batch as tuple/list, got {type(batch)}")
        n = len(batch)
        if n == 4:
            s, c, d, y = batch
            return s, c, d, y, None, None, None, None, None, None
        if n == 6:
            s, c, d, y, Tmin, Tmax = batch
            return s, c, d, y, Tmin, Tmax, None, None, None, None
        if n == 8:
            s, c, d, y, sp, cp, dp, yp = batch
            return s, c, d, y, None, None, sp, cp, dp, yp
        if n == 10:
            s, c, d, y, Tmin, Tmax, sp, cp, dp, yp = batch
            return s, c, d, y, Tmin, Tmax, sp, cp, dp, yp
        raise ValueError(
            f"Batch has {n} elements; expected 4, 6, 8, or 10."
        )

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 200,
        early_stopping_patience: int = 30,
        warmup_epochs: int = 5,
        lambda_ramp_epochs: int = 15,
        verbose: bool = True,
        dataset=None,
        train_indices=None,
    ):
        print(f"\nStarting RAMC Training: {self.trainer_name}")
        if self.alpha_rollout > 0.0:
            print(f"  Rollout-aware fidelity ENABLED: alpha={self.alpha_rollout}, H_r={self.rollout_horizon}")
        print("-" * 70)

        warmup_epochs = int(warmup_epochs)
        lambda_ramp_epochs = int(lambda_ramp_epochs)

        if self.target_lambda == 0.0:
            selection_start_epoch = 0
        else:
            selection_start_epoch = warmup_epochs + max(0, lambda_ramp_epochs - 1)

        print(f"  selection_start_epoch: {selection_start_epoch}")

        # Create rollout sampler if rollout fidelity is enabled
        rollout_sampler = None
        if self.alpha_rollout > 0.0:
            if dataset is None or train_indices is None:
                print(
                    "  WARNING: alpha_rollout > 0 but dataset/train_indices not provided. "
                    "Rollout fidelity disabled for this run."
                )
            else:
                try:
                    rollout_sampler = RolloutSequenceSampler(
                        dataset=dataset,
                        train_indices=train_indices,
                        rollout_horizon=self.rollout_horizon,
                        batch_size=self.rollout_batch_size,
                    )
                except RuntimeError as e:
                    print(f"  WARNING: RolloutSequenceSampler creation failed: {e}")
                    print("  Continuing without rollout fidelity.")

        start_time = time.time()

        for epoch in range(int(num_epochs)):
            # Lambda scheduling
            if self.target_lambda == 0.0:
                self.current_lambda = 0.0
            elif epoch < warmup_epochs:
                self.current_lambda = 0.0
            elif epoch < warmup_epochs + lambda_ramp_epochs:
                progress = (epoch - warmup_epochs + 1) / float(max(1, lambda_ramp_epochs))
                self.current_lambda = self.target_lambda * float(progress)
            else:
                self.current_lambda = self.target_lambda

            train_metrics = self.train_epoch(
                train_loader,
                current_lambda=self.current_lambda,
                rollout_sampler=rollout_sampler,
            )
            val_metrics = self.validate(val_loader, current_lambda=self.current_lambda)

            val_total_target = float(
                val_metrics["fidelity_loss"] + self.target_lambda * val_metrics["risk_loss"]
            )

            if epoch < selection_start_epoch:
                self.scheduler.step(val_metrics["fidelity_loss"])
            else:
                self.scheduler.step(val_total_target)

            current_lr = float(self.optimizer.param_groups[0]["lr"])

            # History logging
            self.history["epoch"].append(int(epoch))
            self.history["learning_rate"].append(current_lr)
            self.history["current_lambda"].append(float(self.current_lambda))

            self.history["train_loss"].append(float(train_metrics["total_loss"]))
            self.history["val_loss"].append(float(val_metrics["total_loss"]))

            self.history["train_total_loss"].append(float(train_metrics["total_loss"]))
            self.history["train_fidelity_loss"].append(float(train_metrics["fidelity_loss"]))
            self.history["train_mse_raw"].append(float(train_metrics.get("mse_raw", 0)))
            self.history["train_risk_loss"].append(float(train_metrics["risk_loss"]))
            self.history["train_expected_cost"].append(float(train_metrics["expected_cost"]))
            self.history["train_rollout_loss"].append(float(train_metrics.get("rollout_loss", 0.0)))

            self.history["val_total_loss"].append(float(val_metrics["total_loss"]))
            self.history["val_total_loss_target"].append(val_total_target)
            self.history["val_fidelity_loss"].append(float(val_metrics["fidelity_loss"]))
            self.history["val_mse_raw"].append(float(val_metrics.get("mse_raw", 0)))
            self.history["val_risk_loss"].append(float(val_metrics["risk_loss"]))
            self.history["val_risk_comfort_loss"].append(float(val_metrics.get("risk_comfort_loss", 0)))
            self.history["val_risk_energy_loss"].append(float(val_metrics.get("risk_energy_loss", 0)))
            self.history["val_expected_cost"].append(float(val_metrics["expected_cost"]))
            self.history["val_expected_comfort"].append(float(val_metrics.get("expected_comfort", 0)))
            self.history["val_expected_energy"].append(float(val_metrics.get("expected_energy_cost", 0)))
            self.history["val_t_air_rmse"].append(float(val_metrics["t_air_rmse"]))

            # Checkpoint selection
            if epoch >= selection_start_epoch:
                combined_metric = val_total_target

                if combined_metric < self.best_val_loss:
                    self.best_val_loss = float(combined_metric)
                    self.best_epoch = int(epoch)
                    self.epochs_without_improvement = 0

                    model_path = os.path.join(self.save_dir, f"{self.trainer_name}_best.pth")
                    self.save_model(model_path, epoch, combined_metric)
                else:
                    self.epochs_without_improvement += 1

                if val_metrics["expected_cost"] < self.best_cost_val:
                    self.best_cost_val = float(val_metrics["expected_cost"])
                    self.best_cost_epoch = int(epoch)
                    best_cost_path = os.path.join(self.save_dir, f"{self.trainer_name}_best_cost.pth")
                    self.save_model(best_cost_path, epoch, val_metrics["expected_cost"])

                if val_metrics["risk_loss"] < self.best_risk_val:
                    self.best_risk_val = float(val_metrics["risk_loss"])
                    self.best_risk_epoch = int(epoch)
                    best_risk_path = os.path.join(self.save_dir, f"{self.trainer_name}_best_risk.pth")
                    self.save_model(best_risk_path, epoch, val_metrics["risk_loss"])
            else:
                self.epochs_without_improvement = 0

            if verbose and (epoch % 10 == 0 or epoch < 10):
                selecting = "YES" if epoch >= selection_start_epoch else "no"
                rollout_str = (
                    f" | rollout={train_metrics.get('rollout_loss', 0.0):.5f}"
                    if rollout_sampler is not None else ""
                )
                print(
                    f"Epoch {epoch:3d} | lambda={self.current_lambda:.3e} | "
                    f"fid={val_metrics['fidelity_loss']:.5f} | "
                    f"risk={val_metrics['risk_loss']:.5f} | "
                    f"T_air RMSE={val_metrics['t_air_rmse']:.4f}"
                    f"{rollout_str} | sel={selecting}"
                )

            if (epoch >= selection_start_epoch) and (self.epochs_without_improvement >= int(early_stopping_patience)):
                print("\nEarly stopping triggered.")
                break

        dt = time.time() - start_time
        print(f"\nTraining completed in {dt:.1f} seconds")

        return self.history

    def train_epoch(self, train_loader, current_lambda: float, rollout_sampler=None) -> Dict[str, float]:
        self.model.train()

        sums = {
            "total_loss": 0.0,
            "fidelity_loss": 0.0,
            "mse_raw": 0.0,
            "risk_loss": 0.0,
            "expected_cost": 0.0,
            "rollout_loss": 0.0,
            "batches": 0,
        }

        for batch in train_loader:
            (states, controls, disturbances, targets,
             Tmin, Tmax, s_pert, c_pert, d_pert, y_pert) = self._unpack_batch(batch)

            # Option A fix: ignore dataset-embedded bounds so the loss always
            # falls back to comfort_bounds in loss_kwargs (i.e. [20, 22] °C).
            if self.ignore_dataset_bounds:
                Tmin = None
                Tmax = None

            states       = states.to(self.device)
            controls     = controls.to(self.device)
            disturbances = disturbances.to(self.device)
            targets      = targets.to(self.device)
            if Tmin is not None: Tmin = Tmin.to(self.device)
            if Tmax is not None: Tmax = Tmax.to(self.device)
            if s_pert is not None: s_pert = s_pert.to(self.device)
            if c_pert is not None: c_pert = c_pert.to(self.device)
            if d_pert is not None: d_pert = d_pert.to(self.device)
            if y_pert is not None: y_pert = y_pert.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            out = calculate_ramc_loss(
                self.model,
                states,
                controls,
                disturbances,
                targets,
                lambda_risk=float(current_lambda),
                Tmin=Tmin,
                Tmax=Tmax,
                skip_risk_if_lambda_zero=True,
                perturbed_states=s_pert,
                perturbed_controls=c_pert,
                perturbed_disturb=d_pert,
                perturbed_targets=y_pert,
                **self.loss_kwargs,
            )

            loss = out["total_loss"]

            # Rollout-aware fidelity (Section 5.3 of design spec)
            rollout_loss_val = 0.0
            if rollout_sampler is not None and self.alpha_rollout > 0:
                x0, c_seq, d_seq, t_seq = rollout_sampler.sample_batch()
                x0    = x0.to(self.device)
                c_seq = c_seq.to(self.device)
                d_seq = d_seq.to(self.device)
                t_seq = t_seq.to(self.device)

                rollout_loss, _ = compute_rollout_fidelity_loss(
                    self.model,
                    x0,
                    c_seq,
                    d_seq,
                    t_seq,
                    output_std=getattr(self.model, "output_std", None),
                    mse_normalize=self.loss_kwargs.get("mse_normalize", True),
                    fidelity_weights=self.loss_kwargs.get("fidelity_weights", None),
                    step_weight_mode=self.rollout_step_weights,
                )

                loss = loss + self.alpha_rollout * rollout_loss
                rollout_loss_val = float(rollout_loss.detach().item())

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip_norm))

            self.optimizer.step()

            sums["total_loss"] += float(out["total_loss"].detach().item())
            sums["fidelity_loss"] += float(out["fidelity_loss"].detach().item())
            sums["mse_raw"] += float(out["mse_raw"].detach().item())

            risk_val = out["risk_loss"]
            sums["risk_loss"] += float(risk_val.item() if torch.is_tensor(risk_val) else risk_val)
            sums["expected_cost"] += float(out["expected_cost"].detach().item())
            sums["rollout_loss"] += rollout_loss_val
            sums["batches"] += 1

        b = max(1, int(sums["batches"]))
        return {
            "total_loss": sums["total_loss"] / b,
            "fidelity_loss": sums["fidelity_loss"] / b,
            "mse_raw": sums["mse_raw"] / b,
            "risk_loss": sums["risk_loss"] / b,
            "expected_cost": sums["expected_cost"] / b,
            "rollout_loss": sums["rollout_loss"] / b,
        }

    def validate(self, val_loader, current_lambda: float) -> Dict[str, float]:
        self.model.eval()

        total_samples = 0
        sums = {
            "total_loss": 0.0,
            "fidelity_loss": 0.0,
            "mse_raw": 0.0,
            "risk_loss": 0.0,
            "risk_comfort_loss": 0.0,
            "risk_energy_loss": 0.0,
            "expected_cost": 0.0,
            "expected_comfort": 0.0,
            "expected_energy_cost": 0.0,
            "t_air_mse": 0.0,
        }

        with torch.no_grad():
            for batch in val_loader:
                (states, controls, disturbances, targets,
                 Tmin, Tmax, s_pert, c_pert, d_pert, y_pert) = self._unpack_batch(batch)

                # Option A fix: ignore dataset-embedded bounds so validation
                # metrics are computed against the same [20, 22] °C target as
                # Phase 3, making val_risk a meaningful predictor of deployment
                # performance.
                if self.ignore_dataset_bounds:
                    Tmin = None
                    Tmax = None

                states       = states.to(self.device)
                controls     = controls.to(self.device)
                disturbances = disturbances.to(self.device)
                targets      = targets.to(self.device)
                if Tmin is not None: Tmin = Tmin.to(self.device)
                if Tmax is not None: Tmax = Tmax.to(self.device)
                if s_pert is not None: s_pert = s_pert.to(self.device)
                if c_pert is not None: c_pert = c_pert.to(self.device)
                if d_pert is not None: d_pert = d_pert.to(self.device)
                if y_pert is not None: y_pert = y_pert.to(self.device)

                bsz = int(states.size(0))
                total_samples += bsz

                out = calculate_ramc_loss(
                    self.model,
                    states,
                    controls,
                    disturbances,
                    targets,
                    lambda_risk=float(current_lambda),
                    Tmin=Tmin,
                    Tmax=Tmax,
                    skip_risk_if_lambda_zero=True,
                    perturbed_states=s_pert,
                    perturbed_controls=c_pert,
                    perturbed_disturb=d_pert,
                    perturbed_targets=y_pert,
                    **self.loss_kwargs,
                )

                sums["total_loss"] += float(out["total_loss"].detach().item()) * bsz
                sums["fidelity_loss"] += float(out["fidelity_loss"].detach().item()) * bsz
                sums["mse_raw"] += float(out["mse_raw"].detach().item()) * bsz
                
                for key in ["risk_loss", "risk_comfort_loss", "risk_energy_loss"]:
                    val = out[key]
                    sums[key] += float(val.item() if torch.is_tensor(val) else val) * bsz
                
                sums["expected_cost"] += float(out["expected_cost"].detach().item()) * bsz
                sums["expected_comfort"] += float(out["expected_comfort"].detach().item()) * bsz
                sums["expected_energy_cost"] += float(out["expected_energy_cost"].detach().item()) * bsz

                preds = self.model(states, controls, disturbances)
                t_air_mse = torch.mean((preds[:, 0] - targets[:, 0]).pow(2))
                sums["t_air_mse"] += float(t_air_mse.item()) * bsz

        denom = max(1, int(total_samples))
        return {
            "total_loss": sums["total_loss"] / denom,
            "fidelity_loss": sums["fidelity_loss"] / denom,
            "mse_raw": sums["mse_raw"] / denom,
            "risk_loss": sums["risk_loss"] / denom,
            "risk_comfort_loss": sums["risk_comfort_loss"] / denom,
            "risk_energy_loss": sums["risk_energy_loss"] / denom,
            "expected_cost": sums["expected_cost"] / denom,
            "expected_comfort": sums["expected_comfort"] / denom,
            "expected_energy_cost": sums["expected_energy_cost"] / denom,
            "t_air_rmse": float(np.sqrt(sums["t_air_mse"] / denom)),
        }


# ---------------------------------------------------------------------
# MSE baseline trainer
# ---------------------------------------------------------------------

class MSETrainer(BaseTrainer):
    """P1: Raw MSE baseline trainer."""

    def __init__(
        self,
        model: ThermalDynamicsNet,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        save_dir: str = "models",
        grad_clip_norm: Optional[float] = 1.0,
        weight_decay: float = 1e-5,
        fidelity_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            device=device,
            save_dir=save_dir,
            weight_decay=weight_decay,
        )
        self.trainer_name = "Raw_MSE_Baseline"
        self.criterion = nn.MSELoss()
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None
        self.fidelity_weights = fidelity_weights

        print(f"MSE Baseline Trainer initialized: {self.trainer_name}")

    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_fidelity = 0.0
        num_batches = 0

        for batch in train_loader:
            states, controls, disturbances, targets = batch[:4]
            states = states.to(self.device)
            controls = controls.to(self.device)
            disturbances = disturbances.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            preds = self.model(states, controls, disturbances)
            loss = self.criterion(preds, targets)

            output_std = getattr(self.model, "output_std", None)
            fid_loss, _ = compute_fidelity_loss(
                preds, targets,
                output_std=output_std,
                mse_normalize=getattr(self.model, "normalization_computed", False),
                fidelity_weights=self.fidelity_weights,
            )

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip_norm))

            self.optimizer.step()

            epoch_loss += float(loss.item())
            epoch_fidelity += float(fid_loss.item())
            num_batches += 1

        return {
            "mse_raw": epoch_loss / max(1, num_batches),
            "fidelity_loss": epoch_fidelity / max(1, num_batches),
        }

    def validate(self, val_loader) -> Dict[str, float]:
        self.model.eval()

        val_loss = 0.0
        val_fidelity = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                states, controls, disturbances, targets = batch[:4]
                states = states.to(self.device)
                controls = controls.to(self.device)
                disturbances = disturbances.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(states, controls, disturbances)
                loss = self.criterion(preds, targets)

                output_std = getattr(self.model, "output_std", None)
                fid_loss, _ = compute_fidelity_loss(
                    preds, targets,
                    output_std=output_std,
                    mse_normalize=getattr(self.model, "normalization_computed", False),
                    fidelity_weights=self.fidelity_weights,
                )

                val_loss += float(loss.item())
                val_fidelity += float(fid_loss.item())
                num_batches += 1

                all_preds.append(preds)
                all_targets.append(targets)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        t_air_rmse = float(torch.sqrt(torch.mean((all_preds[:, 0] - all_targets[:, 0]) ** 2)).item())

        return {
            "mse_raw": val_loss / max(1, num_batches),
            "fidelity_loss": val_fidelity / max(1, num_batches),
            "t_air_rmse": t_air_rmse,
        }

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 200,
        early_stopping_patience: int = 30,
        verbose: bool = True,
    ):
        print(f"\nStarting MSE Baseline Training: {self.trainer_name}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(int(num_epochs)):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics["mse_raw"])

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.history["epoch"].append(int(epoch))
            self.history["learning_rate"].append(current_lr)
            self.history["train_loss"].append(float(train_metrics["mse_raw"]))
            self.history["val_loss"].append(float(val_metrics["mse_raw"]))

            if val_metrics["mse_raw"] < self.best_val_loss:
                self.best_val_loss = float(val_metrics["mse_raw"])
                self.best_epoch = int(epoch)
                self.epochs_without_improvement = 0

                model_path = os.path.join(self.save_dir, f"{self.trainer_name}_best.pth")
                self.save_model(model_path, epoch, val_metrics["mse_raw"])
            else:
                self.epochs_without_improvement += 1

            if verbose and (epoch % 10 == 0 or epoch < 10):
                print(
                    f"Epoch {epoch:3d} | mse_raw={val_metrics['mse_raw']:.6f} | "
                    f"fidelity={val_metrics['fidelity_loss']:.6f} | "
                    f"T_air RMSE={val_metrics['t_air_rmse']:.4f}"
                )

            if self.epochs_without_improvement >= int(early_stopping_patience):
                print("\nEarly stopping triggered.")
                break

        dt = time.time() - start_time
        print(f"\nMSE training completed in {dt:.1f} seconds")

        return self.history


# ---------------------------------------------------------------------
# Training manager
# ---------------------------------------------------------------------

class TrainingManager:
    """Manages training of multiple models with P1-P10 improvements."""

    def __init__(
        self,
        device: str = "cpu",
        save_dir: str = "comparative_models",
        model_config: Optional[dict] = None,
    ):
        self.device = device
        self.save_dir = save_dir
        self.model_config = model_config or {}

        os.makedirs(save_dir, exist_ok=True)

        self.results: Dict[str, Dict[str, Any]] = {}

        self.mse_trainer: Optional[MSETrainer] = None
        self.fidelity_trainer: Optional[RAMCTrainer] = None
        self.ramc_trainers: Dict[float, RAMCTrainer] = {}

    def train_all_baselines(
        self,
        train_loader,
        val_loader,
        lambda_values: List[float],
        num_epochs: int,
        *,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 30,
        warmup_epochs: int = 5,
        lambda_ramp_epochs: int = 15,
        grad_clip_norm: Optional[float] = 1.0,
        weight_decay: float = 1e-5,
        loss_params: Optional[dict] = None,
        include_mse_baseline: bool = True,
        include_fidelity_only: bool = True,
        include_ablations: bool = False,
        ablation_lambda: Optional[float] = None,
        verbose: bool = True,
        dataset=None,
        train_indices=None,
    ):
        print("Starting RAMC training study")
        print("=" * 80)

        loss_params = dict(loss_params or {})
        fidelity_weights = loss_params.get("fidelity_weights", None)

        model_configs = []

        if include_mse_baseline:
            model_configs.append({
                "name": "Raw_MSE_Baseline",
                "trainer_class": MSETrainer,
                "params": {"fidelity_weights": fidelity_weights},
            })

        if include_fidelity_only:
            model_configs.append({
                "name": "Fidelity_Baseline",
                "trainer_class": RAMCTrainer,
                "params": {"lambda_risk": 0.0},
            })

        for lam in lambda_values:
            lam = float(lam)
            if include_fidelity_only and lam == 0.0:
                continue
            model_configs.append({
                "name": f"RAMC_lambda_{lam}",
                "trainer_class": RAMCTrainer,
                "params": {"lambda_risk": lam},
            })
            
        if include_ablations:
            abl_lam = (
                float(ablation_lambda) if ablation_lambda is not None
                else (float(lambda_values[-1]) if lambda_values else 1.5e-3)
            )
            # A1: perturbation-only, RC-plant grounded (uses pre-generated labels)
            model_configs.append({
                "name": f"PertOnly_gamma_{abl_lam}",
                "trainer_class": RAMCTrainer,
                "params": {"lambda_risk": abl_lam, "loss_mode": "pert_only"},
            })
            # A2: mean operational cost (same machinery as RAMC, no tail focus)
            model_configs.append({
                "name": f"MeanCost_lambda_{abl_lam}",
                "trainer_class": RAMCTrainer,
                "params": {"lambda_risk": abl_lam, "risk_operator": "mean"},
            })

        for config in model_configs:
            name = config["name"]
            print(f"\nTraining: {name}")

            model = ThermalDynamicsNet(**self.model_config)

            if hasattr(model, "compute_normalization"):
                model.compute_normalization(train_loader)

            trainer_params = dict(config["params"])
            trainer_params.update({
                "learning_rate": float(learning_rate),
                "device": self.device,
                "save_dir": self.save_dir,
                "grad_clip_norm": grad_clip_norm,
                "weight_decay": float(weight_decay),
            })

            if config["trainer_class"] is RAMCTrainer:
                merged = dict(loss_params)
                merged.update(trainer_params)   # per-model overrides win
                trainer_params = merged

            trainer = config["trainer_class"](model=model, **trainer_params)

            if isinstance(trainer, RAMCTrainer):
                history = trainer.train(
                    train_loader,
                    val_loader,
                    num_epochs=int(num_epochs),
                    early_stopping_patience=int(early_stopping_patience),
                    warmup_epochs=int(warmup_epochs),
                    lambda_ramp_epochs=int(lambda_ramp_epochs),
                    verbose=bool(verbose),
                    dataset=dataset,
                    train_indices=train_indices,
                )
            else:
                history = trainer.train(
                    train_loader,
                    val_loader,
                    num_epochs=int(num_epochs),
                    early_stopping_patience=int(early_stopping_patience),
                    verbose=bool(verbose),
                )

            self.results[name] = {
                "trainer": trainer,
                "history": history,
                "best_val_loss": float(trainer.best_val_loss),
                "config": config,
            }

            if name == "Raw_MSE_Baseline":
                self.mse_trainer = trainer
            elif name == "Fidelity_Baseline":
                self.fidelity_trainer = trainer
            elif name.startswith("RAMC_lambda_"):
                lam = float(config["params"]["lambda_risk"])
                self.ramc_trainers[lam] = trainer
            elif name.startswith("PertOnly_") or name.startswith("MeanCost_"):
                if not hasattr(self, "ablation_trainers"):
                    self.ablation_trainers = {}
                self.ablation_trainers[name] = trainer

            print(f"Completed: {name}")

        print(f"\nAll training completed. Results saved to: {self.save_dir}")
        return self.results

    def save_results_summary(self, filepath: str = "training_results_summary.json"):
        summary = {}
        for name, result in self.results.items():
            trainer = result["trainer"]
            summary[name] = {
                "best_val_loss": float(result["best_val_loss"]),
                "best_epoch": int(trainer.best_epoch),
                "final_epoch": int(len(result["history"].get("epoch", [])) - 1),
                "config": result.get("config", {}).get("params", {}),
            }

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results summary saved to: {filepath}")

    @staticmethod
    def _set_seed(seed: int):
        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _mean_ci(values: List[float]) -> Tuple[float, float]:
        if len(values) <= 1:
            return float(values[0]) if values else float("nan"), 0.0
        arr = np.asarray(values, dtype=np.float64)
        m = float(np.mean(arr))
        ci = float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))
        return m, ci

    @staticmethod
    def _lambda_label(lam: float, all_lambdas: List[float]) -> str:
        """Qualitative label for lambda value."""
        if len(all_lambdas) <= 1:
            return f"={lam:.1e}"
        rank = sorted(all_lambdas).index(lam)
        n = len(all_lambdas)
        if rank == 0:
            qual = "low"
        elif rank == n - 1:
            qual = "high"
        elif rank < n / 3:
            qual = "low"
        elif rank < 2 * n / 3:
            qual = "mid"
        else:
            qual = "high"
        return f"={lam:.1e} ({qual})"

    def analyze_tradeoff(
        self,
        loader,
        loss_config: dict,
        save_path: Optional[str] = None,
        selection_policy: str = "combined",
        seeds: Optional[List[int]] = None,
    ):
        """P1/P10: Analyze with both fidelity_loss and mse_raw, decomposed risk."""
        print(f"Analyzing trade-off (selection_policy={selection_policy})")

        seeds = seeds or [42, 43, 44, 45, 46]

        eval_config = dict(loss_config)
        eval_config["num_perturbations"] = int(eval_config.get("num_perturbations", 100))
        eval_config["lambda_risk"] = float(eval_config.get("lambda_risk", 1.0))

        models: List[Tuple[str, float, BaseTrainer]] = []
        if self.mse_trainer is not None:
            models.append(("Raw MSE Baseline", 0.0, self.mse_trainer))
        if self.fidelity_trainer is not None:
            models.append(("Fidelity Baseline", 0.0, self.fidelity_trainer))
        for lam in sorted(self.ramc_trainers.keys()):
            models.append((f"RAMC lambda={lam}", float(lam), self.ramc_trainers[lam]))

        results = []

        for model_name, lam, trainer in models:
            if selection_policy == "combined":
                suffix = "_best.pth"
            elif selection_policy == "cost":
                suffix = "_best_cost.pth"
            elif selection_policy == "risk":
                suffix = "_best_risk.pth"
            else:
                raise ValueError("selection_policy must be: combined, cost, risk")

            ckpt = os.path.join(self.save_dir, f"{trainer.trainer_name}{suffix}")
            if not os.path.exists(ckpt):
                print(f"Missing checkpoint for {model_name}: {ckpt}")
                continue

            trainer.load_model(ckpt)

            metrics_list = {
                "risk": [], "risk_comfort": [], "risk_energy": [],
                "cost": [], "comfort": [], "energy": [],
                "mse_raw": [], "fidelity": [], "rmse": [],
                "Q_rmse": [], "Q_mae": [], "Q_bias": [],
            }

            for s in seeds:
                self._set_seed(s)

                m = evaluate_on_loader(
                    trainer.model, loader,
                    loss_config=eval_config,
                    device=self.device,
                    compute_occupancy_split=True,
                )

                metrics_list["risk"].append(float(m["risk_loss"]))
                metrics_list["risk_comfort"].append(float(m["risk_comfort_loss"]))
                metrics_list["risk_energy"].append(float(m["risk_energy_loss"]))
                metrics_list["cost"].append(float(m["expected_cost"]))
                metrics_list["comfort"].append(float(m["expected_comfort"]))
                metrics_list["energy"].append(float(m["expected_energy_cost"]))
                metrics_list["mse_raw"].append(float(m["mse_raw"]))
                metrics_list["fidelity"].append(float(m["fidelity_loss"]))
                metrics_list["rmse"].append(float(m["t_air_rmse"]))
                metrics_list["Q_rmse"].append(float(m["Q_rmse"]))
                metrics_list["Q_mae"].append(float(m["Q_mae"]))
                metrics_list["Q_bias"].append(float(m["Q_bias"]))

            result_row = {
                "model_name": model_name,
                "lambda": float(lam),
                "trainer": trainer,
            }
            
            for key, vals in metrics_list.items():
                mean, ci = self._mean_ci(vals)
                result_row[f"{key}_mean"] = float(mean)
                result_row[f"{key}_ci"] = float(ci)

            results.append(result_row)

        self._print_results_table_v2(results, selection_policy)
        
        if save_path:
            self._plot_tradeoff_v2(results, save_path, selection_policy, eval_config)
        
        return results

    def _print_results_table_v2(self, results, selection_policy):
        print(f"\nDETAILED RESULTS TABLE ({selection_policy.upper()} SELECTION)")
        print("=" * 180)
        print(f"{'Model':<22} {'λ':<8} {'mse_raw':<18} {'fidelity':<18} {'risk':<18} "
              f"{'risk_comfort':<18} {'risk_energy':<18} {'Q_rmse':<14}")
        print("-" * 180)

        for r in results:
            print(
                f"{r['model_name']:<22} {r['lambda']:<8.4f} "
                f"{r['mse_raw_mean']:.5f}±{r['mse_raw_ci']:.5f}   "
                f"{r['fidelity_mean']:.5f}±{r['fidelity_ci']:.5f}   "
                f"{r['risk_mean']:.5f}±{r['risk_ci']:.5f}   "
                f"{r['risk_comfort_mean']:.5f}±{r['risk_comfort_ci']:.5f}   "
                f"{r['risk_energy_mean']:.5f}±{r['risk_energy_ci']:.5f}   "
                f"{r['Q_rmse_mean']:.2f}±{r['Q_rmse_ci']:.2f}"
            )

        print("-" * 180)

    def _plot_tradeoff_v2(self, results, save_path, selection_policy, eval_config):
        """Generate trade-off plots matching RAMC Journal Proposal Fig. 5."""
        
        # Changed from 2x3 to 2x4 to accommodate new panels
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
        mse_data = [r for r in results if "MSE" in r["model_name"]]
        fid_data = [r for r in results if "Fidelity" in r["model_name"]]
        ramc_data = [r for r in results if r["model_name"].startswith("RAMC")]
        ramc_data.sort(key=lambda x: x["lambda"])
    
        lambdas = [r["lambda"] for r in ramc_data] if ramc_data else []
    
        # Plot 1: Risk vs Lambda (existing)
        ax = axes[0, 0]
        if ramc_data:
            ax.errorbar(lambdas, [r["risk_mean"] for r in ramc_data],
                       yerr=[r["risk_ci"] for r in ramc_data], fmt="o-", capsize=4, label="RAMC")
            ax.set_xscale("log")
        for data, label in [(mse_data, "Raw MSE"), (fid_data, "Fidelity")]:
            if data:
                ax.axhline(data[0]["risk_mean"], linestyle="--", alpha=0.7, label=label)
        ax.set_xlabel("λ (risk weight)")
        ax.set_ylabel("Worst-10% total cost (CVaR₀.₉)")
        ax.set_title("(a) Tail cost vs λ")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        # Plot 2: Fidelity vs Lambda (existing)
        ax = axes[0, 1]
        if ramc_data:
            ax.errorbar(lambdas, [r["fidelity_mean"] for r in ramc_data],
                       yerr=[r["fidelity_ci"] for r in ramc_data],
                       fmt="o-", capsize=4, label="Fidelity Loss")
            ax.errorbar(lambdas, [r["mse_raw_mean"] for r in ramc_data],
                       yerr=[r["mse_raw_ci"] for r in ramc_data],
                       fmt="s--", capsize=4, alpha=0.7, label="Raw MSE")
            ax.set_xscale("log")
        for data, label in [(mse_data, "Raw MSE baseline"), (fid_data, "Fidelity baseline")]:
            if data:
                ax.axhline(data[0]["fidelity_mean"], linestyle="--", alpha=0.5, label=f"{label}")
        ax.set_xlabel("λ (risk weight)")
        ax.set_ylabel("Loss")
        ax.set_title("(b) Prediction fidelity vs λ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
        # Plot 3: Expected Cost vs Lambda (NEW - required by proposal)
        ax = axes[0, 2]
        if ramc_data:
            ax.errorbar(lambdas, [r["cost_mean"] for r in ramc_data],
                       yerr=[r["cost_ci"] for r in ramc_data],
                       fmt="o-", capsize=4, label="RAMC", color="green")
            ax.set_xscale("log")
        for data, label in [(mse_data, "Raw MSE"), (fid_data, "Fidelity")]:
            if data:
                ax.axhline(data[0]["cost_mean"], linestyle="--", alpha=0.7, label=label)
        ax.set_xlabel("λ (risk weight)")
        ax.set_ylabel("Mean one-step cost")
        ax.set_title("(c) Expected cost vs λ")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        # Plot 4: Risk vs Expected Cost (NEW - required by proposal)
        ax = axes[0, 3]
        if ramc_data:
            ax.errorbar([r["cost_mean"] for r in ramc_data],
                       [r["risk_mean"] for r in ramc_data],
                       xerr=[r["cost_ci"] for r in ramc_data],
                       yerr=[r["risk_ci"] for r in ramc_data],
                       fmt="o-", capsize=3, alpha=0.8, label="RAMC Frontier", color="purple")
            # Change 3: Add qualitative lambda labels to panel (d)
            all_lams = [r["lambda"] for r in ramc_data]
            for r in ramc_data:
                label = self._lambda_label(r["lambda"], all_lams)
                ax.annotate(label, (r["cost_mean"], r["risk_mean"]),
                           xytext=(5, 5), textcoords="offset points", fontsize=7)
        if mse_data:
            r = mse_data[0]
            ax.errorbar(r["cost_mean"], r["risk_mean"],
                       xerr=r["cost_ci"], yerr=r["risk_ci"],
                       fmt="s", markersize=10, capsize=4, label="Raw MSE", color="orange")
        if fid_data:
            r = fid_data[0]
            ax.errorbar(r["cost_mean"], r["risk_mean"],
                       xerr=r["cost_ci"], yerr=r["risk_ci"],
                       fmt="^", markersize=10, capsize=4, label="Fidelity", color="green")
        ax.set_xlabel("Mean one-step cost")
        ax.set_ylabel("Worst-10% total cost (CVaR₀.₉)")
        ax.set_title("(d) Tail cost vs expected cost")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        # Plot 5: Risk decomposition (existing, moved)
        ax = axes[1, 0]
        if ramc_data:
            ax.errorbar(lambdas, [r["risk_comfort_mean"] for r in ramc_data],
                       yerr=[r["risk_comfort_ci"] for r in ramc_data],
                       fmt="o-", capsize=4, label="Comfort Risk")
            ax.errorbar(lambdas, [r["risk_energy_mean"] for r in ramc_data],
                       yerr=[r["risk_energy_ci"] for r in ramc_data],
                       fmt="s-", capsize=4, label="Energy Risk")
            ax.set_xscale("log")
        ax.set_xlabel("λ (risk weight)")
        ax.set_ylabel("Worst-10% cost component (CVaR₀.₉)")
        ax.set_title("(e) Risk decomposition")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        # Plot 6: T_air RMSE (existing, moved)
        ax = axes[1, 1]
        if ramc_data:
            ax.errorbar(lambdas, [r["rmse_mean"] for r in ramc_data],
                       yerr=[r["rmse_ci"] for r in ramc_data],
                       fmt="o-", capsize=4, label="RAMC")
            ax.set_xscale("log")
        for data, label in [(mse_data, "Raw MSE"), (fid_data, "Fidelity")]:
            if data:
                ax.axhline(data[0]["rmse_mean"], linestyle="--", alpha=0.7, label=label)
        ax.set_xlabel("λ (risk weight)")
        ax.set_ylabel("T_air RMSE (°C)")
        ax.set_title("(f) Air temperature prediction error")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        # Plot 7: Q_rmse (existing, moved)
        ax = axes[1, 2]
        if ramc_data:
            ax.errorbar(lambdas, [r["Q_rmse_mean"] for r in ramc_data],
                       yerr=[r["Q_rmse_ci"] for r in ramc_data],
                       fmt="o-", capsize=4, label="RAMC")
            ax.set_xscale("log")
        for data, label in [(mse_data, "Raw MSE"), (fid_data, "Fidelity")]:
            if data:
                ax.axhline(data[0]["Q_rmse_mean"], linestyle="--", alpha=0.7, label=label)
        ax.set_xlabel("λ (risk weight)")
        ax.set_ylabel("Q RMSE (W)")
        ax.set_title("(g) Heating power proxy error")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        # Plot 8: Pareto frontier Risk vs Fidelity (existing, moved)
        ax = axes[1, 3]
        if ramc_data:
            ax.errorbar([r["risk_mean"] for r in ramc_data],
                       [r["fidelity_mean"] for r in ramc_data],
                       xerr=[r["risk_ci"] for r in ramc_data],
                       yerr=[r["fidelity_ci"] for r in ramc_data],
                       fmt="o-", capsize=3, alpha=0.8, label="RAMC Frontier")
            # Change 3: Add qualitative lambda labels to panel (h)
            all_lams = [r["lambda"] for r in ramc_data]
            for r in ramc_data:
                label = self._lambda_label(r["lambda"], all_lams)
                ax.annotate(label, (r["risk_mean"], r["fidelity_mean"]),
                           xytext=(5, 5), textcoords="offset points", fontsize=7)
        if mse_data:
            r = mse_data[0]
            ax.errorbar(r["risk_mean"], r["fidelity_mean"],
                       xerr=r["risk_ci"], yerr=r["fidelity_ci"],
                       fmt="s", markersize=10, capsize=4, label="Raw MSE")
        if fid_data:
            r = fid_data[0]
            ax.errorbar(r["risk_mean"], r["fidelity_mean"],
                       xerr=r["risk_ci"], yerr=r["fidelity_ci"],
                       fmt="^", markersize=10, capsize=4, label="Fidelity")
        ax.set_xlabel("Worst-10% total cost (CVaR₀.₉)")
        ax.set_ylabel("Weighted prediction loss")
        ax.set_title("(h) Trade-off: tail cost vs fidelity")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        # Title with CVaR alpha
        alpha = eval_config.get('cvar_alpha', 0.9)
        title = f"RAMC Open-Loop Analysis - {selection_policy.title()} Selection"
        title += f" (K={eval_config.get('num_perturbations')}, CVaR α={alpha})"
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Trade-off plot saved: {save_path}")


    def analyze_per_state_fidelity(self, loader, save_path: Optional[str] = None):
        print("Analyzing per-state fidelity...")
    
        state_names = ["T_air", "T_env", "T_int", "T_rad1", "T_rad2", "T_ret"]
        results = []
    
        all_models: List[Tuple[str, BaseTrainer]] = []
        if self.mse_trainer is not None:
            all_models.append(("Raw MSE Baseline", self.mse_trainer))
        if self.fidelity_trainer is not None:
            all_models.append(("Fidelity Baseline", self.fidelity_trainer))
        for lam in sorted(self.ramc_trainers.keys()):
            all_models.append((f"RAMC λ={lam}", self.ramc_trainers[lam]))
    
        for model_name, trainer in all_models:
            ckpt_path = os.path.join(self.save_dir, f"{trainer.trainer_name}_best.pth")
            if not os.path.exists(ckpt_path):
                continue
    
            trainer.load_model(ckpt_path)
            trainer.model.eval()
    
            all_preds = []
            all_targets = []
    
            with torch.no_grad():
                for batch in loader:
                    states, controls, disturbances, targets = batch[:4]
                    states = states.to(self.device)
                    controls = controls.to(self.device)
                    disturbances = disturbances.to(self.device)
                    targets = targets.to(self.device)
    
                    preds = trainer.model(states, controls, disturbances)
                    all_preds.append(preds)
                    all_targets.append(targets)
    
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
    
            per_state_rmse = torch.sqrt(torch.mean((all_preds - all_targets).pow(2), dim=0))
            t_air_bias = float((all_preds[:, 0].mean() - all_targets[:, 0].mean()).item())
    
            row = {"model_name": model_name, "t_air_bias": t_air_bias}
            for i, sn in enumerate(state_names):
                row[f"{sn}_rmse"] = float(per_state_rmse[i].item())
            results.append(row)
    
        self._plot_per_state_fidelity(results, state_names, save_path)
        return results

    def _plot_per_state_fidelity(self, results, state_names, save_path):
        if not results:
            print("No fidelity results to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        model_names = [r["model_name"] for r in results]
        x_pos = np.arange(len(model_names))

        bar_width = 0.12
        colors = plt.cm.Set3(np.linspace(0, 1, len(state_names)))

        for i, state in enumerate(state_names):
            rmse_values = [r[f"{state}_rmse"] for r in results]
            offset = (i - len(state_names) / 2) * bar_width
            ax1.bar(x_pos + offset, rmse_values, bar_width, label=state, color=colors[i], alpha=0.85)

        ax1.set_xlabel("Model")
        ax1.set_ylabel("RMSE (°C)")
        ax1.set_title("Per-State RMSE")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha="right")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        biases = [r["t_air_bias"] for r in results]
        colors_bias = ["red" if abs(b) > 0.3 else "green" for b in biases]

        ax2.bar(x_pos, biases, color=colors_bias, alpha=0.75)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.7)
        ax2.axhline(0.3, color="orange", linestyle="--", alpha=0.6, label="Warning threshold")
        ax2.axhline(-0.3, color="orange", linestyle="--", alpha=0.6)

        ax2.set_xlabel("Model")
        ax2.set_ylabel("T_air Bias (°C)")
        ax2.set_title("T_air Prediction Bias")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def analyze_pareto_frontier(
        self,
        loader,
        loss_config: dict,
        save_path: Optional[str] = None,
        selection_policy: str = "combined",
    ):
        print("Analyzing Pareto frontier...")

        results = self.analyze_tradeoff(
            loader, loss_config=loss_config, save_path=None, selection_policy=selection_policy,
        )

        if not results:
            print("No results to plot.")
            return results

        mse_data = [r for r in results if "MSE" in r["model_name"]]
        fid_data = [r for r in results if "Fidelity" in r["model_name"]]
        ramc_data = [r for r in results if r["model_name"].startswith("RAMC")]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if mse_data:
            r = mse_data[0]
            ax.errorbar(r["risk_mean"], r["rmse_mean"], xerr=r["risk_ci"], yerr=r["rmse_ci"],
                       fmt="s", markersize=10, capsize=4, label="Raw MSE Baseline")

        if fid_data:
            r = fid_data[0]
            ax.errorbar(r["risk_mean"], r["rmse_mean"], xerr=r["risk_ci"], yerr=r["rmse_ci"],
                       fmt="^", markersize=10, capsize=4, label="Fidelity Baseline")

        if ramc_data:
            xs = [r["risk_mean"] for r in ramc_data]
            ys = [r["rmse_mean"] for r in ramc_data]
            xerr = [r["risk_ci"] for r in ramc_data]
            yerr = [r["rmse_ci"] for r in ramc_data]
            labels = [r["lambda"] for r in ramc_data]

            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o", capsize=3, alpha=0.8, label="RAMC")

            order = np.argsort(xs)
            ax.plot([xs[i] for i in order], [ys[i] for i in order], "--", linewidth=1, alpha=0.5)

            for x, y, lam in zip(xs, ys, labels):
                ax.annotate(f"λ={lam}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

        ax.set_xlabel("Risk Loss")
        ax.set_ylabel("T_air RMSE (°C)")
        ax.set_title("Pareto Frontier: Risk vs T_air Fidelity")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return results
    
    @staticmethod
    def export_per_state_fidelity_csv(results: List[Dict], save_path: str):
        """Export per-state RMSE results to CSV for Table IV."""
        import pandas as pd
        
        rows = []
        for r in results:
            row = {"Model": r["model_name"], "T_air_bias": r["t_air_bias"]}
            for state in ["T_air", "T_env", "T_int", "T_rad1", "T_rad2", "T_ret"]:
                row[f"{state}_RMSE"] = r.get(f"{state}_rmse", float('nan'))
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"Per-state fidelity table saved: {save_path}")
        return df
    
    @staticmethod
    def export_bias_table_csv(bias_results: List[Dict], save_path: str):
        """Export T_air bias results to CSV for Table V."""
        import pandas as pd
        
        rows = []
        for r in bias_results:
            status = "OK" if abs(r["t_air_bias"]) < 0.3 else "WARNING"
            rows.append({
                "Model": r["model"],
                "T_air_Bias_C": r["t_air_bias"],
                "Status": status
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"Bias table saved: {save_path}")
        return df


if __name__ == "__main__":
    pass