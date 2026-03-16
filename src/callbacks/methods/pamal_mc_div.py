"""
PaMaL-MC-Div: PaMaL + Mixed-Curvature + Diversity Regularization.

Extends PaMaL with:
1. Mixed-Curvature encoder/decoder blocks inserted between the shared encoder and task decoders
2. Magnitude diversity regularization computed on MC embeddings across rays (Option B)
"""

import logging
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.factory.mixed_curvature_layers import MixedCurvatureBlock

from .pamal import PaMaL
from .utils.samplers import Sampler

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class PaMaLMCDiv(PaMaL):
    """PaMaL with Mixed-Curvature processing and diversity regularization.

    Architecture:
        LeNet (subspace by PaMaL) → ★ L_diverse → MC Encoder → MC Decoder → Task Decoders
    
    The MC blocks use stereographic projection (geoopt) with learnable curvatures.
    Diversity is measured via magnitude (effective distinct points) on encoder embeddings across rays.
    """

    def __init__(
        self,
        num_tasks: int,
        ray_sampler: Sampler,
        num: int,
        reg_coefficient: float,
        diversity_coefficient: float = 0.1,
        num_subspaces: int = 5,
        embed_dim: int = 50,
        reweight_lr=True,
        reinit_flag=True,
        **kwargs,
    ):
        super().__init__(
            num_tasks=num_tasks,
            ray_sampler=ray_sampler,
            num=num,
            reg_coefficient=reg_coefficient,
            reweight_lr=reweight_lr,
            reinit_flag=reinit_flag,
            **kwargs,
        )
        self.diversity_coefficient = diversity_coefficient
        self.num_subspaces = num_subspaces
        self.embed_dim = embed_dim

        logging.info(
            f"PaMaL-MC-Div: diversity_coeff={diversity_coefficient}, "
            f"num_subspaces={num_subspaces}, embed_dim={embed_dim}"
        )

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        """Configure model: apply PaMaL subspace + attach MC encoder block."""
        # 1. Apply PaMaL subspace conversion (LeNet + decoders)
        super().configure_model(model, data_module, keep_original_weights)

        # 2. Attach MC encoder block to the model (no MC decoder)
        model.mc_encoder_block = MixedCurvatureBlock(
            dim=self.embed_dim, num_subspaces=self.num_subspaces
        )

        # Prevent PaMaL's make_subspace_compatible from touching MC block
        model.mc_encoder_block._skip_lorafy = True

        # Storage for embeddings across rays (for diversity loss)
        model._mc_embeddings = []

        logging.info("PaMaL-MC-Div: Attached MC encoder block to model (no decoder)")
        return model

    def compute_diversity_loss(self, trainer: "BaseTrainer") -> torch.Tensor:
        """Compute magnitude-based diversity loss on MC embeddings across rays.
        
        Inspired by magnipy (https://github.com/aidos-lab/magnipy):
        For each sample in the batch, compute the "magnitude" (effective number 
        of distinct points) across its ray embeddings:
        
        1. Pairwise distances between ray embeddings: D_ij = ||z_ray_i - z_ray_j||
        2. Similarity matrix: Z_ij = exp(-t * D_ij)
        3. Solve Z * w = 1 for magnitude weights w
        4. Magnitude = sum(w)
        
        Higher magnitude = more diverse/distinct embeddings across rays.
        Loss = -magnitude (minimize to maximize diversity).
        
        The matrix Z is only (num_rays x num_rays), so the solve is very fast.
        """
        model = trainer.model
        mc_embeddings = getattr(model, '_mc_embeddings', [])

        if len(mc_embeddings) < 2:
            return torch.tensor(0.0, device=trainer.device, requires_grad=True)

        num_rays = len(mc_embeddings)
        
        # Stack embeddings: (num_rays, B, D) -> (B, num_rays, D)
        Z = torch.stack(mc_embeddings, dim=0).permute(1, 0, 2)
        B = Z.shape[0]
        
        # Pairwise distances between ray embeddings for each sample: (B, num_rays, num_rays)
        dists = torch.cdist(Z, Z, p=2)
        
        # Similarity matrix: S_ij = exp(-t * d_ij), with scale t=1.0
        # Adding small epsilon to diagonal for numerical stability
        S = torch.exp(-dists) + 1e-6 * torch.eye(num_rays, device=Z.device).unsqueeze(0)
        
        # Solve S * w = 1 for magnitude weights
        ones = torch.ones(B, num_rays, 1, device=Z.device)
        w = torch.linalg.solve(S, ones)  # (B, num_rays, 1)
        
        # Magnitude = sum of weights per sample
        magnitude = w.sum(dim=1).squeeze(-1)  # (B,)
        
        # Average magnitude across batch
        avg_magnitude = magnitude.mean()
        
        # Maximize magnitude = minimize negative magnitude
        return -avg_magnitude

    def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        """Add diversity regularization to training loss."""
        if self.diversity_coefficient > 0:
            div_loss = self.compute_diversity_loss(trainer)
            trainer.loss += self.diversity_coefficient * div_loss

            # Log diversity loss
            try:
                import wandb
                wandb.log({
                    "diversity_loss": div_loss.item(),
                    "mystep": trainer.current_step,
                })
            except (ImportError, wandb.Error):
                pass

        # Clear stored MC embeddings for next training step
        if hasattr(trainer.model, '_mc_embeddings'):
            trainer.model._mc_embeddings = []

    def get_weighted_loss(
        self,
        losses: Tensor,
        ray: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        """Same as PaMaL, but losses already include MC processing."""
        return super().get_weighted_loss(losses=losses, ray=ray, **kwargs)
