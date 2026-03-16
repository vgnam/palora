"""
PaMaL-QD: PaMaL + Soft Quality-Diversity with Momentum Target Encoder.

Two-stream architecture:
  Stream 1 (Online): Standard PaMaL forward → task losses + weighted loss
  Stream 2 (Target): Frozen EMA encoder → stable behavior embeddings for QD

Algorithm (per training step):
  1. Sample W α-vectors from Dir(p)                    [PaMaL]
  2. Materialize: x_i = α_iᵀΘ                          [PaMaL]
  3. Forward online encoder: ŷ_i = model(x_i; ray_i)   → task losses L_i
  4. Forward target encoder: z⁻_i = enc⁻(x_i; ray_i)   → stable embeddings (no grad)
  5. Quality: f_i = exp(-L̄_i)  (WITH gradient through L_i)
  6. Soft QD: A = Σ f_i − Σ_{i<j} √(f_i·f_j) · K(z⁻_i, z⁻_j)
  7. L_total = PaMaL_loss + γ · (-A)
  8. Backprop → gradient flows through f_i → online encoder
  9. EMA update: φ⁻ ← τ·φ + (1−τ)·φ⁻

Gradient analysis:
  ∂(-A)/∂θ = -Σ ∂f_i/∂θ + Σ_{i<j} K_ij · ∂√(f_i·f_j)/∂θ
  where K_ij is CONSTANT (from target encoder) → acts as weight
  → Pushes online encoder to: improve quality + differentiate
    quality when solutions are close in behavior space
"""

import copy
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

import wandb

from .pamal import PaMaL

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class PaMaLQD(PaMaL):
    """PaMaL with Soft Quality-Diversity loss using momentum target encoder.

    Architecture:
        Online:  encoder (subspace) → decoders  [trained normally by PaMaL]
        Target:  encoder (EMA copy, frozen)      [provides stable z for QD kernel]

    The QD loss gradient flows ONLY through the quality scores f_i = exp(-loss_i),
    NOT through the behavior descriptors z⁻_i (which are from the frozen target encoder).
    The kernel K(z⁻_i, z⁻_j) acts as a constant weight that determines which
    pairs of solutions should be encouraged to have different qualities.
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

        self._target_encoder = None  # initialized in configure_model

    # ─── Model Setup ──────────────────────────────────────────────────

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        """Apply PaMaL subspace conversion + create frozen target encoder."""
        # 1. PaMaL subspace conversion (modifies model.encoder in-place)
        super().configure_model(model, data_module, keep_original_weights)

        # 2. Create target encoder = deep copy of online encoder (frozen)
        self._target_encoder = copy.deepcopy(model.encoder)
        for param in self._target_encoder.parameters():
            param.requires_grad = False
        self._target_encoder.eval()

        # 3. Storage for per-ray inputs
        model._qd_inputs = []

        logging.info(
            f"PaMaL-QD: target encoder (EMA τ={self.ema_tau}), "
            f"γ={self.qd_coefficient}, σ²={self.sigma_sq}, "
            f"adaptive_σ={self.adaptive_sigma}"
        )
        return model

    def configure_param_groups(self, model: nn.Module, lr=None):
        """Target encoder is on self (not model) and frozen → not in optimizer."""
        return super().configure_param_groups(model, lr=lr)

    # ─── Target Encoder EMA ──────────────────────────────────────────

    @torch.no_grad()
    def _ema_update_target_encoder(self, online_encoder: nn.Module):
        """EMA update: φ⁻ ← τ·φ + (1−τ)·φ⁻"""
        tau = self.ema_tau
        for p_online, p_target in zip(
            online_encoder.parameters(), self._target_encoder.parameters()
        ):
            p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    # ─── Soft QD Loss (SQUAD Paper) ──────────────────────────────────

    def compute_softqd_loss(
        self,
        embeddings: torch.Tensor,
        qualities: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Soft QD loss (SQUAD paper, Eq. 1).

        A(θ) = Σ f_i  −  Σ_{i<j} √(f_i · f_j) · K(z⁻_i, z⁻_j)

        where:
          f_i = quality of solution i        (WITH grad → drives diversity)
          z⁻_i = target encoder embedding    (detached → stable behavior space)
          K(z⁻_i, z⁻_j) = exp(-||z⁻_i - z⁻_j||² / σ²)

        Args:
            embeddings: (W, D) target encoder embeddings (detached, no grad)
            qualities: (W,) quality scores f_i = exp(-loss_i) (WITH grad!)

        Returns:
            Scalar loss = -A (to be minimized by optimizer)
        """
        W = embeddings.shape[0]
        if W < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        f = qualities.clamp(min=eps)  # (W,) — HAS gradient

        # ── Kernel from target encoder (constant, no grad) ──
        with torch.no_grad():
            D_sq = torch.cdist(embeddings, embeddings).pow(2)  # (W, W)
            K = torch.exp(-D_sq / self.sigma_sq)               # (W, W)

            # Zero out self-interactions and negligible entries
            K.fill_diagonal_(0.0)
            K = K.masked_fill(K < eps, 0.0)

            # Adaptive σ² using mean nearest-neighbor distance
            if self.adaptive_sigma:
                D_sq_no_self = D_sq.clone()
                D_sq_no_self.fill_diagonal_(float('inf'))
                nn_dists = D_sq_no_self.min(dim=1).values  # (W,)
                self.sigma_sq = nn_dists.mean().clamp(min=eps).item()

        # ── Attractive term: Σ f_i ──
        attraction = f.sum()  # gradient through f

        # ── Repulsive term: Σ_{i<j} √(f_i · f_j) · K_ij ──
        # K_ij is constant (detached), gradient flows through √(f_i·f_j)
        f_outer = f.unsqueeze(0) * f.unsqueeze(1)          # (W, W) — has grad
        f_sqrt_prod = torch.sqrt(f_outer.clamp(min=eps))   # (W, W) — has grad
        repulsion = torch.triu(K * f_sqrt_prod, diagonal=1).sum()  # scalar — has grad

        # ── Objective A = attraction - repulsion ──
        A = attraction - repulsion

        return -A  # minimize -A = maximize A

    # ─── Training Hooks ──────────────────────────────────────────────

    def on_before_forward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Store input batch for target encoder forward (called per ray)."""
        super().on_before_forward(trainer, *args, **kwargs)
        if self.qd_coefficient > 0 and trainer.model.training:
            if not hasattr(trainer.model, '_qd_inputs'):
                trainer.model._qd_inputs = []
            # Store (x, ray) for target encoder forward in on_before_backward
            trainer.model._qd_inputs.append(
                (trainer.x.detach(), trainer.method.ray.detach())
            )

    def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Compute Soft QD loss and add to trainer.loss.

        Steps:
          1. Forward stored inputs through target encoder (no grad) → embeddings
          2. Compute quality f_i = exp(-avg_loss_i) WITH gradient
          3. Compute A and add γ·(-A) to total loss
        """
        super().on_before_backward(trainer, *args, **kwargs)

        if self.qd_coefficient <= 0 or not trainer.model.training:
            return

        model = trainer.model
        if not hasattr(model, '_qd_inputs') or len(model._qd_inputs) == 0:
            return

        # ── Step 4: Forward target encoder (no grad) → stable embeddings ──
        target_embeddings = []
        with torch.no_grad():
            self._target_encoder.eval()
            for x, ray in model._qd_inputs:
                z = self._target_encoder(x, ray=ray)  # (B, D)
                target_embeddings.append(z.mean(dim=0))  # (D,) mean-pool over batch
        model._qd_inputs = []

        if len(target_embeddings) < 2:
            return

        z_stack = torch.stack(target_embeddings)  # (W, D) — detached

        # ── Step 5: Quality scores f_i = exp(-avg_loss_i) WITH gradient ──
        # trainer.losses is list[dict], each dict: task_name → loss_tensor
        # DO NOT detach: gradient flows through loss → f_i → QD loss → backprop
        qualities = []
        for loss_dict in trainer.losses:
            avg_loss = sum(loss_dict.values()) / len(loss_dict)
            qualities.append(torch.exp(-avg_loss))  # NO .detach()!
        qualities = torch.stack(qualities)  # (W,) — has grad

        # ── Step 6: Soft QD loss ──
        qd_loss = self.compute_softqd_loss(z_stack, qualities)

        # ── Step 7: Add to total loss ──
        trainer.loss = trainer.loss + self.qd_coefficient * qd_loss

        # Logging
        trainer.qd_loss = qd_loss.item()
        try:
            wandb.log({
                "qd_loss": qd_loss.item(),
                "sigma_sq": self.sigma_sq,
                "mystep": trainer.current_step,
            })
        except Exception:
            pass

    # ─── Step 9: EMA Update ──────────────────────────────────────────

    def on_after_optimizer_step(self, trainer: "BaseTrainer", *args, **kwargs):
        """EMA update target encoder after each optimization step."""
        super().on_after_optimizer_step(trainer, *args, **kwargs)
        if self._target_encoder is not None:
            self._ema_update_target_encoder(trainer.model.encoder)

    def connect(self, trainer: "BaseTrainer"):
        """Move target encoder to device when trainer connects."""
        super().connect(trainer)
        if self._target_encoder is not None:
            self._target_encoder = self._target_encoder.to(trainer.device)
