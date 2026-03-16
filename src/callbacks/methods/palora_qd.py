"""
PaLoRA-QD: PaLoRA + Soft Quality-Diversity (SQUAD Paper).

Same Soft QD approach as PaMaL-QD but using PaLoRA (LoRA adapters) as base.
QD loss computed on online encoder embeddings so gradients flow through
both fitness (quality) and behavior descriptor (diversity) terms.
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

import wandb

from .palora import PaLoRA

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class PaLoRAQD(PaLoRA):
    """PaLoRA with Soft Quality-Diversity loss (SQUAD paper).

    Uses online encoder embeddings (with gradient) for the QD loss.
    Adaptive σ² via nearest-neighbor distance.
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
        self.sigma_sq = sigma ** 2
        self.ema_tau = ema_tau
        self.adaptive_sigma = adaptive_sigma

        self._ray_embeddings = []
        self._ray_losses = []

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        """Configure model: apply PaLoRA LoRA adapters."""
        super().configure_model(model, data_module, keep_original_weights)
        logging.info(
            f"PaLoRA-QD: QD coeff={self.qd_coefficient}, σ²={self.sigma_sq}, "
            f"adaptive_σ={self.adaptive_sigma}"
        )
        return model

    def compute_softqd_loss(
        self,
        embeddings: torch.Tensor,
        qualities: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Soft QD loss (SQUAD paper).

        A(θ) = Σ f_i  −  Σ_{i<j} √(f_i · f_j) · K(b_i, b_j)
        K = exp(-||b_i - b_j||² / σ²)

        Returns -A (minimize to maximize archive quality + diversity).
        """
        W = embeddings.shape[0]
        if W < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        f = qualities.clamp(min=eps)
        D_sq = torch.cdist(embeddings, embeddings).pow(2)
        K = torch.exp(-D_sq / self.sigma_sq)

        mask = torch.eye(W, device=K.device, dtype=torch.bool)
        K = K.masked_fill(mask, 0.0)
        K = K.masked_fill(K < eps, 0.0)

        f_sqrt_prod = torch.sqrt(f.unsqueeze(0) * f.unsqueeze(1))
        repulsion = torch.triu(K * f_sqrt_prod, diagonal=1).sum()
        A = f.sum() - repulsion

        if self.adaptive_sigma:
            with torch.no_grad():
                D_sq_no_self = D_sq.clone()
                D_sq_no_self.fill_diagonal_(float('inf'))
                nn_dists = D_sq_no_self.min(dim=1).values
                self.sigma_sq = nn_dists.mean().clamp(min=eps).item()

        return -A

    def on_before_forward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Clear storage on first ray of training step."""
        super().on_before_forward(trainer, *args, **kwargs)
        if self.qd_coefficient <= 0 or not trainer.model.training:
            return
        if len(trainer.losses) == 0:
            self._ray_embeddings = []
            self._ray_losses = []

    def on_after_forward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Store online encoder embedding WITH gradient."""
        super().on_after_forward(trainer, *args, **kwargs)
        if self.qd_coefficient <= 0 or not trainer.model.training:
            return
        if hasattr(trainer, 'features') and trainer.features is not None:
            embedding = trainer.features.mean(dim=0)
            self._ray_embeddings.append(embedding)

    def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Add PaLoRA cosine loss + Soft QD loss."""
        # PaLoRA's cosine loss
        super().on_before_backward(trainer, *args, **kwargs)

        if self.qd_coefficient <= 0 or not trainer.model.training:
            return
        if len(self._ray_embeddings) < 2:
            return

        z_stack = torch.stack(self._ray_embeddings)

        qualities = []
        for loss_dict in trainer.losses:
            avg_loss = sum(loss_dict.values()) / len(loss_dict)
            qualities.append(torch.exp(-avg_loss).detach())
        qualities = torch.stack(qualities)

        qd_loss = self.compute_softqd_loss(z_stack, qualities)
        trainer.loss = trainer.loss + self.qd_coefficient * qd_loss

        trainer.qd_loss = qd_loss.item()
        try:
            wandb.log({
                "qd_loss": qd_loss.item(),
                "sigma_sq": self.sigma_sq,
                "mystep": trainer.current_step,
            })
        except Exception:
            pass

        self._ray_embeddings = []
        self._ray_losses = []
