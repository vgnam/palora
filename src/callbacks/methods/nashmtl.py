from typing import Dict, List, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback


class NashMTL(AlgoCallback):
    def __init__(
        self,
        num_tasks: int,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
    ):
        super().__init__(num_tasks)
        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.num_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.num_tasks, dtype=np.float32)

    def __repr__(self) -> str:
        return "NashMTL(max_norm={}, update_weights_every={}, optim_niter={})".format(
            self.max_norm, self.update_weights_every, self.optim_niter
        )

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.num_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.num_tasks,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.num_tasks, self.num_tasks), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.num_tasks):
            constraint.append(-cp.log(self.alpha_param[i] * self.normalization_factor_param) - cp.log(G_alpha[i]) <= 0)
        obj = cp.Minimize(cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param)
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        extra_outputs = dict()
        if self.step == 0:
            self._init_optim_problem()

        if isinstance(losses, dict):
            losses = torch.stack(tuple(losses.values()))

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            grads = {}
            for i, loss in enumerate(losses):
                g = list(torch.autograd.grad(loss, shared_parameters, retain_graph=True))
                grad = torch.cat([torch.flatten(grad) for grad in g])
                grads[i] = grad

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.step += 1
            alpha = self.prvs_alpha

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        return weighted_loss, extra_outputs

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[Parameter], torch.Tensor] = None,
        representation: Union[List[Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            **kwargs,
        )
        loss.backward()

        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        return loss, extra_outputs
