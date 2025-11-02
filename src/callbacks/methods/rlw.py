import torch

from .algo_callback import AlgoCallback
import torch.nn.functional as F


class RandomLossWeighting(AlgoCallback):
    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.num_tasks
        weight = (F.softmax(torch.randn(self.num_tasks), dim=-1)).to(self.device)
        if isinstance(losses, dict):
            losses = torch.stack(list(losses.values()), dim=0)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)
