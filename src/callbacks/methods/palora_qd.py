"""
PaLoRA-QD: PaLoRA + Soft Quality-Diversity.

Same Soft QD approach as PaMaL-QD but using PaLoRA (LoRA adapters) as base.
Soft QD loss is computed on ONLINE encoder embeddings (with gradients)
so that the QD gradient flows through both:
  - behavior descriptors b(θ): pushes solutions apart in latent space
  - quality scores f(θ): pushes solutions toward lower task loss
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
    """PaLoRA with Soft Quality-Diversity loss.

    Following the SQUAD paper, the QD loss gradient flows through BOTH
    the behavior descriptors (encoder embeddings) and quality scores,
    so the encoder is directly optimized for diversity.
    """

    def __init__(
        self,
        qd_coefficient: float = 0.1,
        sigma: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.qd_coefficient = qd_coefficient
        self.sigma = sigma

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        """Configure model: apply PaLoRA LoRA adapters + init QD storage."""
        super().configure_model(model, data_module, keep_original_weights)
        model._qd_inputs = []
        logging.info(
            f"PaLoRA-QD: QD coeff={self.qd_coefficient}, σ={self.sigma}"
        )
        return model

    def compute_softqd_loss(
        self, embeddings: torch.Tensor, qualities: torch.Tensor
    ) -> torch.Tensor:
        """Compute Soft QD loss (SQUAD paper).

        A(θ) = Σ f_i  −  Σ_{i<j} √(f_i · f_j) · K(b_i, b_j)
        K(b_i, b_j) = exp(-||b_i - b_j||² / σ²)

        Both embeddings and qualities MUST have gradients.

        Returns -A (minimize → maximize archive quality).
        """
        W = embeddings.shape[0]
        if W < 2:
            return torch.tensor(0.0, device=embeddings.device)

        f = qualities.clamp(min=1e-8)
        D_sq = torch.cdist(embeddings, embeddings).pow(2)
        K = torch.exp(-D_sq / (self.sigma ** 2))
        f_sqrt_prod = torch.sqrt(f.unsqueeze(0) * f.unsqueeze(1))
        repulsion = torch.triu(K * f_sqrt_prod, diagonal=1).sum()
        A = f.sum() - repulsion
        return -A

    def on_before_forward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Store input batch and ray for QD loss computation."""
        super().on_before_forward(trainer, *args, **kwargs)
        if self.qd_coefficient > 0 and trainer.model.training:
            if not hasattr(trainer.model, '_qd_inputs'):
                trainer.model._qd_inputs = []
            trainer.model._qd_inputs.append(
                (trainer.x.detach(), trainer.method.ray.detach())
            )

    def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Compute Soft QD loss using ONLINE encoder (with gradients).
        Also adds PaLoRA's cosine loss via super().
        """
        super().on_before_backward(trainer, *args, **kwargs)

        if self.qd_coefficient <= 0 or not trainer.model.training:
            return

        model = trainer.model
        if not hasattr(model, '_qd_inputs') or len(model._qd_inputs) == 0:
            return

        # Re-forward through online encoder (WITH gradients)
        online_embeddings = []
        for x, ray in model._qd_inputs:
            z = model.encoder(x, ray=ray)       # (B, D) — HAS grad_fn
            online_embeddings.append(z.mean(dim=0))

        if len(online_embeddings) < 2:
            model._qd_inputs = []
            return

        z_stack = torch.stack(online_embeddings)  # (W, D)

        # Quality scores (WITH gradients)
        qualities = []
        for loss_dict in trainer.losses:
            avg_loss = sum(loss_dict.values()) / len(loss_dict)
            qualities.append(torch.exp(-avg_loss))  # NO .detach()
        qualities = torch.stack(qualities)

        # Soft QD loss
        qd_loss = self.compute_softqd_loss(z_stack, qualities)
        trainer.loss += self.qd_coefficient * qd_loss

        trainer.qd_loss = qd_loss.item()
        wandb.log({"qd_loss": qd_loss.item(), "mystep": trainer.current_step})

        model._qd_inputs = []
