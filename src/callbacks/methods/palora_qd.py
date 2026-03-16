"""
PaLoRA-QD: PaLoRA + Soft Quality-Diversity with Momentum Target Encoder.

Same two-stream design as PaMaL-QD but using PaLoRA (LoRA adapters) as base.

Algorithm (per training step):
  1. PaLoRA samples W rays, forward through online model → task losses
  2. Forward target encoder (EMA, frozen) → stable behavior embeddings
  3. Quality: f_i = exp(-avg_loss_i) WITH gradient
  4. Soft QD: A = Σ f_i − Σ_{i<j} √(f_i·f_j) · K(z⁻_i, z⁻_j)
  5. L_total = PaLoRA_loss + cosine_loss + γ · (-A)
  6. Backprop (gradient flows through f_i to online encoder)
  7. EMA update: φ⁻ ← τ·φ + (1−τ)·φ⁻
"""

import copy
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

import wandb

from .palora import PaLoRA

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class PaLoRAQD(PaLoRA):
    """PaLoRA with Soft Quality-Diversity loss using momentum target encoder."""

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
        self._target_encoder = None

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        """Apply PaLoRA LoRA conversion + create frozen target encoder."""
        super().configure_model(model, data_module, keep_original_weights)

        self._target_encoder = copy.deepcopy(model.encoder)
        for param in self._target_encoder.parameters():
            param.requires_grad = False
        self._target_encoder.eval()

        model._qd_inputs = []

        logging.info(
            f"PaLoRA-QD: target encoder (EMA τ={self.ema_tau}), "
            f"γ={self.qd_coefficient}, σ²={self.sigma_sq}"
        )
        return model

    @torch.no_grad()
    def _ema_update_target_encoder(self, online_encoder: nn.Module):
        """EMA update: φ⁻ ← τ·φ + (1−τ)·φ⁻"""
        tau = self.ema_tau
        for p_online, p_target in zip(
            online_encoder.parameters(), self._target_encoder.parameters()
        ):
            p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    def compute_softqd_loss(
        self, embeddings: torch.Tensor, qualities: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Soft QD loss: A = Σf_i − Σ_{i<j} √(f_i·f_j)·K_ij, return -A."""
        W = embeddings.shape[0]
        if W < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        f = qualities.clamp(min=eps)

        with torch.no_grad():
            D_sq = torch.cdist(embeddings, embeddings).pow(2)
            K = torch.exp(-D_sq / self.sigma_sq)
            K.fill_diagonal_(0.0)
            K = K.masked_fill(K < eps, 0.0)

            if self.adaptive_sigma:
                D_sq_no_self = D_sq.clone()
                D_sq_no_self.fill_diagonal_(float('inf'))
                nn_dists = D_sq_no_self.min(dim=1).values
                self.sigma_sq = nn_dists.mean().clamp(min=eps).item()

        attraction = f.sum()
        f_outer = f.unsqueeze(0) * f.unsqueeze(1)
        f_sqrt_prod = torch.sqrt(f_outer.clamp(min=eps))
        repulsion = torch.triu(K * f_sqrt_prod, diagonal=1).sum()
        A = attraction - repulsion
        return -A

    def on_before_forward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Store input batch for target encoder forward (called per ray)."""
        super().on_before_forward(trainer, *args, **kwargs)
        if self.qd_coefficient > 0 and trainer.model.training:
            if not hasattr(trainer.model, '_qd_inputs'):
                trainer.model._qd_inputs = []
            trainer.model._qd_inputs.append(
                (trainer.x.detach(), trainer.method.ray.detach())
            )

    def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Add PaLoRA cosine loss + Soft QD loss."""
        super().on_before_backward(trainer, *args, **kwargs)

        if self.qd_coefficient <= 0 or not trainer.model.training:
            return

        model = trainer.model
        if not hasattr(model, '_qd_inputs') or len(model._qd_inputs) == 0:
            return

        # Target encoder forward (no grad)
        target_embeddings = []
        with torch.no_grad():
            self._target_encoder.eval()
            for x, ray in model._qd_inputs:
                z = self._target_encoder(x, ray=ray)
                target_embeddings.append(z.mean(dim=0))
        model._qd_inputs = []

        if len(target_embeddings) < 2:
            return

        z_stack = torch.stack(target_embeddings)

        # Quality WITH gradient (NO .detach())
        qualities = []
        for loss_dict in trainer.losses:
            avg_loss = sum(loss_dict.values()) / len(loss_dict)
            qualities.append(torch.exp(-avg_loss))
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

    def on_after_optimizer_step(self, trainer: "BaseTrainer", *args, **kwargs):
        """EMA update target encoder after each optimization step."""
        super().on_after_optimizer_step(trainer, *args, **kwargs)
        if self._target_encoder is not None:
            self._ema_update_target_encoder(trainer.model.encoder)

    def connect(self, trainer: "BaseTrainer"):
        """Move target encoder to device."""
        super().connect(trainer)
        if self._target_encoder is not None:
            self._target_encoder = self._target_encoder.to(trainer.device)
