"""
PaGeL: Pareto Geometry Learning with Mixed-Curvature Neural Networks.

PaGeL = PaMaL (SubspaceConv with linear weight interpolation via ray)
      + Geometry-aware losses from mixed-curvature latent space.

Based on Theorem 4: linear interpolation αθ + (1-α)θ' is sufficient
to approximate any N continuous mappings.
"""

import logging
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .algo_callback import ParetoFrontApproximationAlgoCallback
from .ll.pagel_modules import (
    GeometryLoss,
    ProductManifoldLatent,
)
from .pamal import PaMaL
from .utils.samplers import Sampler

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class PaGeL(PaMaL):
    """PaGeL = PaMaL + Geometry Loss.

    Inherits everything from PaMaL (SubspaceConv, ray-based interpolation).
    Adds: product manifold latent sampling + geometry-aware losses.
    """

    def __init__(
        self,
        num_tasks: int,
        ray_sampler: Sampler,
        num: int,
        reg_coefficient: float = 0.0,
        # PaGeL-specific
        euclidean_dim: int = 16,
        hyperbolic_dim: int = 16,
        spherical_dim: int = 16,
        lambda_geo: float = 0.01,
        lambda_tangent: float = 1.0,
        lambda_metric: float = 0.1,
        num_subspaces: int = 5,
        learnable_curvature: bool = True,
        init_curvature: float = 0.1,
        geo_warmup_epochs: int = 10,
        reweight_lr: bool = True,
        reinit_flag: bool = True,
        **kwargs,
    ):
        super().__init__(
            num_tasks=num_tasks,
            ray_sampler=ray_sampler,
            num=num,
            reg_coefficient=reg_coefficient,
            reweight_lr=reweight_lr,
            reinit_flag=reinit_flag,
        )

        self.lambda_geo = lambda_geo
        self.geo_warmup_epochs = geo_warmup_epochs

        # Product manifold latent space (for geometry loss)
        self.latent = ProductManifoldLatent(
            euclidean_dim=euclidean_dim,
            hyperbolic_dim=hyperbolic_dim,
            spherical_dim=spherical_dim,
            num_subspaces=num_subspaces,
            learnable_curvature=learnable_curvature,
            init_curvature=init_curvature,
        )

        # Geometry loss
        self.geo_loss_fn = GeometryLoss(
            lambda_tangent=lambda_tangent,
            lambda_metric=lambda_metric,
        )

        # Runtime state
        self._current_z: torch.Tensor = None

        logging.info(
            f"PaGeL: PaMaL + geometry loss "
            f"(eucl={euclidean_dim}, hyp={hyperbolic_dim}, sph={spherical_dim}, "
            f"lambda_geo={lambda_geo}, warmup={geo_warmup_epochs})"
        )

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        """PaMaL's SubspaceConv conversion + attach latent space to model."""
        # PaMaL: convert layers to SubspaceConv
        model = super().configure_model(model, data_module, keep_original_weights)

        # Attach latent to model for device management
        model._pagel_latent = self.latent
        model._pagel_latent._skip_lorafy = True

        return model

    def connect(self, trainer: "BaseTrainer"):
        super().connect(trainer)
        self._trainer = trainer

    def on_before_forward(self, *args, **kwargs):
        """Sample z from product manifold (for geometry loss later)."""
        trainer = self._trainer
        device = trainer.device

        # Move latent to device if needed
        if next(self.latent.parameters()).device != torch.device(device):
            self.latent = self.latent.to(device)
            self.geo_loss_fn = self.geo_loss_fn.to(device)

        # Sample z — only used for geometry loss, not for forward pass
        batch_size = trainer.x.shape[0]
        z = self.latent.sample_uniform(batch_size, device)
        z.requires_grad_(True)
        self._current_z = z

    def on_before_backward(self, *args, **kwargs):
        """Add geometry loss after warmup."""
        trainer = self._trainer

        current_epoch = getattr(trainer, "epoch", 0)
        if current_epoch < self.geo_warmup_epochs:
            self._current_z = None
            return

        if self.lambda_geo > 0 and self._current_z is not None:
            ray = trainer.ray
            if ray is not None:
                try:
                    raw_losses = trainer.losses
                    if isinstance(raw_losses, list):
                        raw_losses = raw_losses[-1]
                    if isinstance(raw_losses, dict):
                        losses = torch.stack(list(raw_losses.values()))
                    else:
                        losses = self.cast_losses_to_correct_type(raw_losses)

                    geo_loss, _ = self.geo_loss_fn(
                        losses=losses,
                        z=self._current_z,
                        ray=ray,
                        latent=self.latent,
                    )
                    trainer.loss = trainer.loss + self.lambda_geo * geo_loss

                except RuntimeError as e:
                    logging.debug(f"PaGeL geometry loss skipped: {e}")

        self._current_z = None
