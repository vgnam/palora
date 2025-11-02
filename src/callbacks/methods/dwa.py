from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback
import numpy as np


class DynamicWeightAverage(AlgoCallback):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(self, num_tasks, iteration_window: int = 25, temp=2.0):
        super().__init__(num_tasks)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, num_tasks), dtype=np.float32)
        self.weights = np.ones(num_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses: Tensor, **kwargs) -> Tuple[Tensor, dict]:
        if isinstance(losses, dict):
            losses = tuple(losses.values())
        if not isinstance(losses, Tensor):
            losses = torch.stack(losses)
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            self.weights = (self.num_tasks * np.exp(ws / self.temp)) / (np.exp(ws / self.temp)).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        loss = (task_weights * losses).mean()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)

    def __repr__(self) -> str:
        return f"DynamicWeightAverage(iteration_window={self.iteration_window}, temp={self.temp})"
