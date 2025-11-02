from collections import defaultdict
from typing import Dict, List

import torch
from src.models.base_model import SharedBottom
from src.utils.moo_losses import MultiForwardRegularizationLoss

from .ensemble_trainer import EnsembleTrainer


class MultiForwardEnsembleTrainer(EnsembleTrainer):
    model: SharedBottom

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num = self.method.num
        self.reg_coefficient = self.method.reg_coefficient

        self.num_tasks = self.benchmark.num_tasks
        self.population_loss = MultiForwardRegularizationLoss(self.num_tasks, self.reg_coefficient)

    def training_step(self):
        return self.training_step1()

    def compute_multiforward_regularization_loss(
        self, losses: List[Dict[str, torch.Tensor]], rays: List[torch.Tensor]
    ):
        task_losses = {}
        for i, loss in enumerate(losses):
            ray = rays[i]
            for j, key in enumerate(loss.keys()):
                if key not in task_losses:
                    task_losses[key] = {}
                task_losses[key][ray[j].item()] = loss[key]

        return self.population_loss(task_losses)

    def training_step1(self):
        # TODO: Repeat the experiments with updated time scale: before we used epoch, now step (more granular)
        rays = self.method.ray_sampler.sample(self.t)
        self.zero_grad_optimizer()
        self.loss = 0

        self.losses = []
        # TODO: this is suboptimal since the forward pass time scales linearly with the number of rays.
        # We should use torch.vmap to parallelize the forward pass, however, torch.func does not currently
        # support torch.nn.maxUnpool2d, which is used in SegNet.
        for i in range(self.num):
            self.ray = rays[i].to(self.device)
            self.method.ray = rays[i].to(self.device)
            self.method.task_weights = rays[i].to(self.device)
            self.on_before_forward()
            self.on_before_forward_callbacks()
            self.y_hat = self.forward(self.ray)
            self.on_after_forward()
            self.on_after_forward_callbacks()

            losses = self.loss_fn(self.y_hat, self.y)
            self.losses.append(losses)

            loss, _ = self.method.get_weighted_loss(
                losses,
                ray=self.ray,
                shared_parameters=list(self.model.shared_parameters()),
                task_specific_parameters=list(self.model.task_specific_parameters()),
                last_shared_parameters=list(self.model.last_shared_parameters()),
                representation=self.features,
                scaler=self.scaler,
            )

            self.loss += loss

            # total_loss += self.loss
            # for i, key in enumerate(self.losses.keys()):
            #     if key not in losses:
            #         losses[key] = {}
            #     losses[key][self.ray[i].item()] = self.losses[key]

        # if self.num > 1 and self.reg_coefficient > 0:
        #     regularization_loss = self.population_loss(losses)
        #     if self.reg_coefficient * regularization_loss < total_loss:
        #         total_loss += self.reg_coefficient * regularization_loss
        self.on_before_backward()
        self.on_before_backward_callbacks()
        self.loss.backward()
        self.on_after_backward()
        self.on_after_backward_callbacks()

        self.on_before_optimizer_step()
        self.on_before_optimizer_step_callbacks()
        self.step_optimizer()
        self.on_after_optimizer_step()
        self.on_after_optimizer_step_callbacks()

        # fix losses to be a dictionary for compatibility with the metric callbacks
        sum_dict = defaultdict(lambda: 0)
        for d in self.losses:
            for key, value in d.items():
                sum_dict[key] += value

        # Convert back to a regular dictionary if needed
        sum_dict = dict(sum_dict)
        self.losses = {key: value / self.num for key, value in sum_dict.items()}
