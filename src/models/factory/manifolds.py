"""
StereographicModel: Wrapper around geoopt's stereographic math for mixed-curvature spaces.

Ported from: lsyysl9711/Mixed_Curvature_VRPs (ICML 2025)
The key idea: split the embedding dimension into P subspaces, each with its own learnable curvature κ.
"""

import torch
from geoopt.manifolds.stereographic import math as gmath


class StereographicModel:
    """Wrapper for stereographic manifold operations on product spaces.
    
    Handles splitting a flat tensor into P subspaces, applying manifold ops per-subspace
    with different curvatures, then merging back.
    """

    def __init__(self):
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def split(self, x, c):
        """Split tensor into P subspaces based on curvature vector length.
        
        Args:
            x: (*, D) tensor
            c: (P,) curvature parameters
        Returns:
            x_split: (*, P, D//P) tensor
            c_expanded: properly shaped curvature for broadcasting
        """
        num_subspaces = c.shape[0]
        D = x.shape[-1]
        others = x.shape[:-1]
        x = x.reshape(*others, num_subspaces, D // num_subspaces)
        c = c.unsqueeze(-1)
        while len(c.shape) < len(x.shape):
            c = c.unsqueeze(0)
        return x, c

    def merge(self, x):
        """Merge P subspaces back into single tensor.
        
        Args:
            x: (*, P, D//P) tensor
        Returns:
            (*, D) tensor
        """
        P, D = x.shape[-2:]
        others = x.shape[:-2]
        return x.reshape(*others, P * D)

    def expmap0(self, u, c):
        """Exponential map from origin: tangent space → manifold."""
        u, c = self.split(u, c)
        return self.merge(gmath.expmap0(u, k=c))

    def logmap0(self, p, c):
        """Logarithmic map to origin: manifold → tangent space."""
        p, c = self.split(p, c)
        return self.merge(gmath.logmap0(p, k=c))

    def proj(self, x, c):
        """Project point onto the manifold (ensure it stays on the manifold)."""
        x, c = self.split(x, c)
        return self.merge(gmath.project(x, k=c))

    def mobius_add(self, x, y, c):
        """Möbius addition on the stereographic model."""
        x, c = self.split(x, c)
        y, _ = self.split(y, c.squeeze(-1).squeeze(0) if c.dim() > 1 else c)
        # Re-split y with same curvature
        c_flat = c.squeeze(-1)
        while c_flat.dim() > 1:
            c_flat = c_flat.squeeze(0)
        y, c2 = self.split(y if y.dim() >= 2 else y, c_flat)
        return self.merge(gmath.mobius_add(x, y, k=c))

    def dist(self, x, y, c):
        """Geodesic distance between two points."""
        return self.sqdist(x, y, c).sqrt()

    def sqdist(self, x, y, c_):
        """Squared geodesic distance."""
        x, c = self.split(x, c_)
        y, _ = self.split(y, c_)

        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = torch.matmul(y.unsqueeze(-2), (-x).unsqueeze(-1)).squeeze(-2)

        denom = 1 - 2 * c * xy + c.pow(2) * x2 * y2
        a = (1 - 2 * c * xy - c * y2) / denom
        b = (1 + c * x2) / denom

        norm = a.pow(2) * x2 + 2 * a * b * xy + b.pow(2) * y2
        dist = 2.0 * gmath.artan_k(norm.clip(min=1e-15).sqrt(), k=c)

        return dist.pow(2).sum(dim=-2)
