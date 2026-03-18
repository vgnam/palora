"""
Pareto HyperNetworks (PHN).

Based on: "Learning the Pareto Front with Hypernetworks" (Navon et al., ICLR 2021)
https://github.com/AvivNavon/pareto-hypernetworks

A hypernetwork takes a preference ray as input and generates all weights
for a target network. The target network then performs the forward pass
using those generated weights.
"""

import logging
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.factory.phn.phn_wrappers import HyperModel
from src.models.factory.phn.solvers import (
    EPOSolver,
    LinearScalarizationSolver,
)

from .algo_callback import ParetoFrontApproximationAlgoCallback
from .utils.samplers import Sampler

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class ParetoHyperNetwork(ParetoFrontApproximationAlgoCallback):
    """Pareto HyperNetworks callback.

    Uses a hypernetwork to generate target network weights conditioned
    on a preference ray vector, enabling single-model Pareto front learning.
    """

    def __init__(
        self,
        num_tasks: int,
        ray_sampler: Sampler,
        solver: str = "linear",
        num: int = 1,
        ray_hidden_dim: int = 100,
        **kwargs,
    ):
        super().__init__(num_tasks=num_tasks, ray_sampler=ray_sampler)
        self.solver_type = solver
        self.num = num
        self.ray_hidden_dim = ray_hidden_dim
        self._solver = None  # initialized after model is configured
        self._trainer = None

        logging.info(
            f"PHN: solver={solver}, ray_hidden_dim={ray_hidden_dim}"
        )

    def configure_model(self, model, data_module=None, **kwargs) -> nn.Module:
        """Replace the SharedBottom model with HyperModel."""
        dataset_name = data_module.name if data_module is not None else "multimnist"
        task_names = data_module.task_names if data_module is not None else None
        hyper_model = HyperModel(dataset_name=dataset_name, task_names=task_names)

        # Initialize solver
        n_params = sum(p.numel() for p in hyper_model.hnet.parameters())
        if self.solver_type == "epo":
            self._solver = EPOSolver(num_tasks=self.num_tasks, n_params=n_params)
        else:
            self._solver = LinearScalarizationSolver(num_tasks=self.num_tasks)

        logging.info(f"PHN: HyperModel created for dataset '{dataset_name}' "
                     f"(hnet params: {n_params:,})")
        return hyper_model

    def connect(self, trainer: "BaseTrainer"):
        super().connect(trainer)
        self._trainer = trainer

    def on_before_forward(self, *args, **kwargs):
        """Set the ray on the model before forward pass."""
        trainer = self._trainer
        if trainer is not None and hasattr(trainer, 'ray') and trainer.ray is not None:
            ray = trainer.ray
            if isinstance(ray, torch.Tensor):
                # Ensure ray is on the correct device and is 1D
                ray = ray.to(trainer.device)
                if ray.dim() > 1:
                    ray = ray.squeeze(0)
            trainer.model.ray = ray

    def get_weighted_loss(
        self,
        losses: Tensor,
        ray: Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        """Compute weighted loss using the configured solver."""
        losses = self.cast_losses_to_correct_type(losses)

        if ray is None:
            ray = self.ray
        if ray is None:
            # Fallback: uniform weighting
            ray = torch.ones(self.num_tasks, device=losses.device) / self.num_tasks

        ray = ray.to(losses.device)
        if ray.dim() > 1:
            ray = ray.squeeze(0)

        # Get hnet parameters for EPO solver
        hnet_params = list(self._trainer.model.hnet.parameters()) if self._trainer is not None else None

        loss = self._solver(losses, ray, hnet_params)
        return loss, {}

    def configure_param_groups(self, model: nn.Module, lr=None):
        """Only the hypernetwork parameters are trainable."""
        if hasattr(model, 'hnet'):
            return [{"params": model.hnet.parameters()}]
        return [{"params": model.parameters()}]

    def parameters(self):
        """PHN has no additional learnable parameters beyond the model."""
        return []
