"""
Mixed-Curvature layers for PaLoRA.

StereographicLinear: Linear layer operating via Möbius matvec in curved space.
MixedCurvatureBlock: Embedding/Encoder block: Exp → StereographicLinear → Log → Residual.
"""

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from geoopt.manifolds.stereographic import math as gmath

from .manifolds import StereographicModel


class StereographicLinear(nn.Module):
    """Linear layer in stereographic (curved) space using Möbius matvec.
    
    Each of the `num_subspaces` subspaces has its own learnable curvature parameter.
    Forward: logmap0 → linear → expmap0 → project (per subspace).
    """

    def __init__(self, manifold, in_features, out_features, num_subspaces, dropout=0.0, use_bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.num_subspaces = num_subspaces
        self.dropout = dropout
        self.use_bias = use_bias

        # Learnable curvatures — initialized to 0 (Euclidean) so training starts stable
        self.c = nn.Parameter(torch.zeros(num_subspaces))

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if use_bias else None

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        if self.bias is not None:
            init.constant_(self.bias, 0)

    def mobius_matvec(self, m, x):
        """Möbius matrix-vector multiplication.
        
        logmap0 → linear in tangent space → expmap0
        """
        curv = self._get_curv()

        x, c = self.manifold.split(x, curv)
        u = self.manifold.merge(gmath.logmap0(x, k=c))
        mu, c = self.manifold.split(u @ m.transpose(-1, -2), curv)
        out = self.manifold.merge(gmath.expmap0(mu, k=c))
        return out

    def _get_curv(self):
        return self.c

    def forward(self, x):
        assert not torch.isnan(x).any(), "NaN in StereographicLinear input"

        curv = self._get_curv()

        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.mobius_matvec(drop_weight, x)
        res = self.manifold.proj(res, curv)

        if self.use_bias and self.bias is not None:
            bias = self.bias.view(1, -1)
            str_bias = self.manifold.expmap0(bias, curv)
            str_bias = self.manifold.proj(str_bias, curv)
            # Möbius addition for bias
            # Simplified: add in tangent space instead of full Möbius add for numerical stability
            res_tangent = self.manifold.logmap0(res, curv)
            bias_tangent = self.manifold.logmap0(str_bias, curv)
            res = self.manifold.expmap0(res_tangent + bias_tangent, curv)
            res = self.manifold.proj(res, curv)

        assert not torch.isnan(res).any(), "NaN in StereographicLinear output"
        return res

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, num_subspaces={self.num_subspaces}"


class MixedCurvatureBlock(nn.Module):
    """One mixed-curvature processing block.
    
    Pattern from the ICML 2025 paper:
        z_curved = logmap0(StereographicLinear(expmap0(z)))
        output = a * z + b * z_curved  (learnable residual weights)
    """

    def __init__(self, dim, num_subspaces):
        super().__init__()
        self.manifold = StereographicModel()
        self.stereo_linear = StereographicLinear(
            self.manifold,
            in_features=dim,
            out_features=dim,
            num_subspaces=num_subspaces,
        )
        # Learnable residual weights (as in the reference implementation)
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, z):
        """
        Args:
            z: (B, D) tensor in Euclidean space
        Returns:
            (B, D) tensor — mixed Euclidean + curved features
        """
        curv = self.stereo_linear.c

        # Map to manifold
        z_manifold = self.manifold.expmap0(z, curv)

        # Linear in curved space
        z_curved = self.stereo_linear(z_manifold)

        # Map back to tangent space
        z_curved = self.manifold.logmap0(z_curved, curv)

        # Learnable residual combination
        return self.a * z + self.b * z_curved
