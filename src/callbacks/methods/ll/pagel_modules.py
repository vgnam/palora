"""
PaGeL Modules: Product Manifold Latent Space and Geometry Loss.

PaGeL = PaMaL + geometry-aware losses from a mixed-curvature latent space.
The latent space Z = E x H x S is used only for computing geometry losses,
not for generating weights (SubspaceConv handles that via ray interpolation).
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds.stereographic import math as gmath

from src.models.factory.manifolds import StereographicModel


class ProductManifoldLatent(nn.Module):
    """Product manifold latent space Z = E x H x S.

    Used for geometry-aware regularization of the Pareto front.
    """

    def __init__(
        self,
        euclidean_dim: int = 16,
        hyperbolic_dim: int = 16,
        spherical_dim: int = 16,
        num_subspaces: int = 1,
        learnable_curvature: bool = True,
        init_curvature: float = 0.1,
    ):
        super().__init__()
        self.euclidean_dim = euclidean_dim
        self.hyperbolic_dim = hyperbolic_dim
        self.spherical_dim = spherical_dim
        self.total_dim = euclidean_dim + hyperbolic_dim + spherical_dim

        self.num_subspaces_hyp = self._valid_subspaces(hyperbolic_dim, num_subspaces)
        self.num_subspaces_sph = self._valid_subspaces(spherical_dim, num_subspaces)

        self.manifold = StereographicModel()

        if learnable_curvature:
            self.c_hyp_raw = nn.Parameter(torch.tensor(math.log(math.expm1(init_curvature))))
            self.c_sph_raw = nn.Parameter(torch.tensor(math.log(math.expm1(init_curvature))))
        else:
            self.register_buffer("c_hyp_raw", torch.tensor(math.log(math.expm1(init_curvature))))
            self.register_buffer("c_sph_raw", torch.tensor(math.log(math.expm1(init_curvature))))

    @staticmethod
    def _valid_subspaces(dim: int, desired: int) -> int:
        if dim == 0:
            return 1
        for n in range(min(desired, dim), 0, -1):
            if dim % n == 0:
                return n
        return 1

    @property
    def c_hyperbolic(self) -> torch.Tensor:
        return -F.softplus(self.c_hyp_raw)

    @property
    def c_spherical(self) -> torch.Tensor:
        return F.softplus(self.c_sph_raw)

    def get_curvatures(self) -> Dict[str, torch.Tensor]:
        return {
            "euclidean": torch.tensor(0.0, device=self.c_hyp_raw.device),
            "hyperbolic": self.c_hyperbolic,
            "spherical": self.c_spherical,
        }

    def _get_hyp_curv_vector(self, device):
        return self.c_hyperbolic.expand(self.num_subspaces_hyp).to(device)

    def _get_sph_curv_vector(self, device):
        return self.c_spherical.expand(self.num_subspaces_sph).to(device)

    def sample_uniform(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample z from product manifold. Returns (B, total_dim)."""
        components = []

        if self.euclidean_dim > 0:
            components.append(torch.randn(batch_size, self.euclidean_dim, device=device) * 0.1)

        if self.hyperbolic_dim > 0:
            v = torch.randn(batch_size, self.hyperbolic_dim, device=device) * 0.1
            c = self._get_hyp_curv_vector(device)
            z = self.manifold.proj(self.manifold.expmap0(v, c), c)
            components.append(z)

        if self.spherical_dim > 0:
            v = torch.randn(batch_size, self.spherical_dim, device=device) * 0.1
            c = self._get_sph_curv_vector(device)
            z = self.manifold.proj(self.manifold.expmap0(v, c), c)
            components.append(z)

        return torch.cat(components, dim=-1)


class GeometryLoss(nn.Module):
    """Geometry-aware losses: tangent alignment + metric preservation."""

    def __init__(self, lambda_tangent: float = 1.0, lambda_metric: float = 0.1):
        super().__init__()
        self.lambda_tangent = lambda_tangent
        self.lambda_metric = lambda_metric

    def forward(
        self,
        losses: torch.Tensor,
        z: torch.Tensor,
        ray: torch.Tensor,
        latent: ProductManifoldLatent,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = losses.device
        m = losses.shape[0]

        # Jacobian J = ∂losses/∂z
        J_rows = []
        for i in range(m):
            grad = torch.autograd.grad(
                losses[i], z, retain_graph=True, create_graph=False, allow_unused=True,
            )
            if grad[0] is not None:
                J_rows.append(grad[0].mean(dim=0))
            else:
                J_rows.append(torch.zeros(z.shape[1], device=device))
        J = torch.stack(J_rows, dim=0)

        # Tangent Alignment: ||J^T · n||^2
        n = ray.to(device)
        L_tangent = (J.T @ n).pow(2).sum()

        # Metric Preservation: ||J^T J / s - I||_F^2
        JtJ = J.T @ J
        d = z.shape[1]
        I = torch.eye(d, device=device)
        scale = JtJ.diagonal().mean().detach().clamp(min=1e-8)
        L_metric = ((JtJ / scale - I) ** 2).sum() / (d * d)

        L_geo = self.lambda_tangent * L_tangent + self.lambda_metric * L_metric

        return L_geo, {
            "tangent_loss": L_tangent.detach(),
            "metric_loss": L_metric.detach(),
            "geo_loss": L_geo.detach(),
        }
