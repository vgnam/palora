from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback
import numpy as np
from scipy.optimize import minimize


class CAGrad(AlgoCallback):
    def __init__(self, num_tasks, c=0.4):
        super().__init__(num_tasks)
        self.c = c

    def __repr__(self) -> str:
        return f"CAGrad(c={self.c})"

    def get_weighted_loss(
        self,
        losses: Tensor,
        shared_parameters: List[Parameter] | Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.num_tasks).to(self.device)

        for i, loss in enumerate(losses):
            if i < self.num_tasks:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.num_tasks) / self.num_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.num_tasks).dot(A).dot(b.reshape(self.num_tasks, 1))
                + c * np.sqrt(x.reshape(1, self.num_tasks).dot(A).dot(x.reshape(self.num_tasks, 1)) + 1e-8)
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g
        elif rescale == 1:
            return g / (1 + alpha**2)
        else:
            return g / (1 + alpha)

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.num_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        if isinstance(losses, dict):
            losses = torch.stack(tuple(losses.values()))

        self.get_weighted_loss(losses, shared_parameters)
        return torch.mean(losses), {}  # NOTE: to align with all other weight methods
