# Adapted from https://github.com/ruchtem/cosmos and https://github.com/AvivNavon/pareto-hypernetworks
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.base_model import BaseModel

from .phn_census import CensusHyper, CensusTarget
from .phn_lenet import LeNetHyper, LeNetTarget
from .phn_resnet import ResnetHyper, ResNetTarget
from .phn_segnet import SegNetHyper, SegNetTarget


class HyperModel(BaseModel):
    def __init__(self, dataset_name, task_names=None):
        super().__init__()
        dataset_name = dataset_name.lower()
        if "census" in dataset_name:
            hnet = CensusHyper(num_tasks=2, ray_hidden_dim=100)
            net = CensusTarget()
        elif dataset_name == "multimnist":
            hnet: nn.Module = LeNetHyper(ray_hidden_dim=100, num_tasks=2)
            net: nn.Module = LeNetTarget(num_tasks=2)
        elif dataset_name == "multimnist3":
            hnet: nn.Module = LeNetHyper(ray_hidden_dim=100, num_tasks=3)
            net: nn.Module = LeNetTarget(num_tasks=3)
        elif dataset_name == "utkface":
            hnet: nn.Module = ResnetHyper(preference_dim=3)
            net: nn.Module = ResNetTarget()
        elif dataset_name == "cityscapes2" or dataset_name == "cityscapes":
            hnet: nn.Module = SegNetHyper()
            net: nn.Module = SegNetTarget()
        else:
            print(dataset_name)
            raise NotImplementedError

        self.hnet = hnet
        self.net = net
        self.ray = None
        self.task_names = task_names

    def forward(self, x, ray=None, return_embedding=False):
        if ray is not None:
            self.ray = ray
        # Ensure ray is on the same device as the model
        if isinstance(self.ray, torch.Tensor):
            device = next(self.hnet.parameters()).device
            self.ray = self.ray.to(device)
            if self.ray.dim() > 1:
                self.ray = self.ray.squeeze(0)
        weights = self.hnet(self.ray)
        outputs = self.net(x, weights)

        # Convert list/tuple outputs to dict keyed by task names
        if isinstance(outputs, (list, tuple)):
            if self.task_names is not None:
                task_outs = {name: out for name, out in zip(self.task_names, outputs)}
            else:
                task_outs = {str(i): out for i, out in enumerate(outputs)}
        elif isinstance(outputs, dict):
            task_outs = outputs
        else:
            task_outs = outputs

        if return_embedding:
            return task_outs, None
        return task_outs

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.hnet.parameters()

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return iter([])

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return iter([])
