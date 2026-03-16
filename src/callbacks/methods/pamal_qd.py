"""
PaMaL-QD: PaMaL + Soft Quality-Diversity (SQUAD Paper).

Integrates the Soft QD objective from Hedayatian & Nikolaidis into PaMaL.
The QD loss is computed on **online encoder** embeddings so that gradients
flow through both the fitness (quality) and behavior descriptor (diversity) terms.

Key difference from naive integration:
  - Online encoder embeddings are used WITH gradients for the QD loss
  - Adaptive σ² via nearest-neighbor distance (as in the paper)
  - Quality = exp(-avg_task_loss), non-negative as required

Paper objective:
  A(θ) = Σ f_i  −  Σ_{i<j} √(f_i · f_j) · K(b_i, b_j)
  K(b_i, b_j) = exp(-||b_i - b_j||² / σ²)

Loss = -A (minimize to maximize archive quality + diversity)
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

import wandb

from .pamal import PaMaL

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class PaMaLQD(PaMaL):
    """PaMaL with Soft Quality-Diversity loss (SQUAD paper).

    Each training step:
      1. PaMaL samples W rays, does W forward passes (online encoder + decoders)
      2. We store per-ray online encoder embeddings (WITH grad)
      3. Before backward, compute Soft QD loss on these embeddings
      4. L_total = PaMaL_loss + γ · L_softqd
      5. Backprop updates encoder to improve both task quality AND diversity

    σ² is adapted each step using mean nearest-neighbor distance in behavior space.
    """

    def __init__(
        self,
        qd_coefficient: float = 0.1,
        sigma: float = 1.0,
        ema_tau: float = 0.01,
        adaptive_sigma: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.qd_coefficient = qd_coefficient
        self.sigma_sq = sigma ** 2  # σ²
        self.ema_tau = ema_tau
        self.adaptive_sigma = adaptive_sigma

        # Storage for per-ray data (populated during forward passes)
        self._ray_embeddings = []  # online encoder embeddings WITH grad
        self._ray_losses = []  # per-ray avg task losses (detached for quality)

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        """Configure model: apply PaMaL subspace conversion."""
        super().configure_model(model, data_module, keep_original_weights)

        logging.info(
            f"PaMaL-QD: QD coeff={self.qd_coefficient}, σ²={self.sigma_sq}, "
            f"adaptive_σ={self.adaptive_sigma}"
        )
        return model

    def compute_softqd_loss(
        self,
        embeddings: torch.Tensor,
        qualities: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Soft QD loss (SQUAD paper, Eq. 1).

        A(θ) = Σ f_i  −  Σ_{i<j} √(f_i · f_j) · K(b_i, b_j)

        where:
          - f_i = quality of solution i (non-negative, detached)
          - b_i = behavior descriptor of solution i (encoder embedding, WITH grad)
          - K(b_i, b_j) = exp(-||b_i - b_j||² / σ²) Gaussian kernel

        Gradients flow through b_i (encoder params) via the kernel term.

        Args:
            embeddings: (W, D) online encoder embeddings per ray (requires_grad=True)
            qualities: (W,) quality scores (detached, non-negative)

        Returns:
            Scalar loss = -A (to be minimized)
        """
        W = embeddings.shape[0]
        if W < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Ensure non-negative fitness
        f = qualities.clamp(min=eps)  # (W,)

        # Pairwise squared distances: ||b_i - b_j||²
        # embeddings has grad → kernel will have grad → loss has grad w.r.t. encoder
        D_sq = torch.cdist(embeddings, embeddings).pow(2)  # (W, W)

        # Gaussian kernel
        K = torch.exp(-D_sq / self.sigma_sq)  # (W, W)

        # Remove self-interactions (diagonal) and negligible entries
        mask = torch.eye(W, device=K.device, dtype=torch.bool)
        K = K.masked_fill(mask, 0.0)
        K = K.masked_fill(K < eps, 0.0)

        # Pairwise fitness product: √(f_i · f_j) — detached (no grad through f)
        f_sqrt_prod = torch.sqrt(f.unsqueeze(0) * f.unsqueeze(1))  # (W, W)

        # Repulsive term: Σ_{i<j} √(f_i·f_j) · K_ij (upper triangle)
        repulsion = torch.triu(K * f_sqrt_prod, diagonal=1).sum()

        # Attractive term: Σ f_i
        attraction = f.sum()

        # Objective A = attraction - repulsion → maximize
        A = attraction - repulsion

        # Adaptive σ² using mean nearest-neighbor distance
        if self.adaptive_sigma:
            with torch.no_grad():
                # D_sq with self-distance set to inf
                D_sq_no_self = D_sq.clone()
                D_sq_no_self.fill_diagonal_(float('inf'))
                nn_dists = D_sq_no_self.min(dim=1).values  # (W,)
                new_sigma_sq = nn_dists.mean().clamp(min=eps).item()
                self.sigma_sq = new_sigma_sq

        return -A

    def on_before_forward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Called before each ray's forward pass. Clear storage on first ray."""
        super().on_before_forward(trainer, *args, **kwargs)

        if self.qd_coefficient <= 0 or not trainer.model.training:
            return

        # On first ray of this training step, clear storage
        # trainer.losses is [] at start, so len == 0 means first ray
        if len(trainer.losses) == 0:
            self._ray_embeddings = []
            self._ray_losses = []

    def on_after_forward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Called after each ray's forward. Store embedding WITH gradient."""
        super().on_after_forward(trainer, *args, **kwargs)

        if self.qd_coefficient <= 0 or not trainer.model.training:
            return

        # trainer.features = encoder output (before decoders), computed in model.forward()
        # This tensor has requires_grad=True since it's part of the computation graph
        if hasattr(trainer, 'features') and trainer.features is not None:
            # Mean pool over batch → one embedding per ray
            embedding = trainer.features.mean(dim=0)  # (D,) — keeps grad
            self._ray_embeddings.append(embedding)

    def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Add Soft QD loss to trainer.loss."""
        super().on_before_backward(trainer, *args, **kwargs)

        if self.qd_coefficient <= 0 or not trainer.model.training:
            return

        if len(self._ray_embeddings) < 2:
            return

        # Stack embeddings — these have grad since they come from online encoder
        z_stack = torch.stack(self._ray_embeddings)  # (W, D)

        # Compute quality scores from per-ray task losses (detached)
        # trainer.losses is list[dict], each dict: task_name → loss_tensor
        qualities = []
        for loss_dict in trainer.losses:
            avg_loss = sum(loss_dict.values()) / len(loss_dict)
            qualities.append(torch.exp(-avg_loss).detach())
        qualities = torch.stack(qualities)  # (W,)

        # Compute Soft QD loss
        qd_loss = self.compute_softqd_loss(z_stack, qualities)
        trainer.loss = trainer.loss + self.qd_coefficient * qd_loss

        # Log
        trainer.qd_loss = qd_loss.item()
        try:
            wandb.log({
                "qd_loss": qd_loss.item(),
                "sigma_sq": self.sigma_sq,
                "mystep": trainer.current_step,
            })
        except Exception:
            pass

        # Clear storage
        self._ray_embeddings = []
        self._ray_losses = []
