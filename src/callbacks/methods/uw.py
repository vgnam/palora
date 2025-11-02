from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback


class UncertaintyWeighting(AlgoCallback):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(self, num_tasks):
        super().__init__(num_tasks)
        self.logsigma = Parameter(torch.zeros((num_tasks,), requires_grad=True))

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum([0.5 * (torch.exp(-logs) * loss + logs) for loss, logs in zip(losses, self.logsigma)])

        return loss, dict(weights=torch.exp(-self.logsigma))  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]
