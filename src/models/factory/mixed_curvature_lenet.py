# """
# MixedCurvatureLeNetR: LeNet encoder enhanced with Mixed-Curvature processing.

# Architecture (following ICML 2025 paper's 3-layer design):
#     1. LeNet feature extraction (existing MultiLeNetR) → 50-dim embedding
#     2. Mixed-Curvature Embedding Layer: Exp → StereographicLinear → Log
#     3. Mixed-Curvature Encoder Layer: Exp → StereographicLinear → Log → Self-Attention → FFN
#     Output: 50-dim embedding (compatible with existing MultiLeNetO decoders)
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .lenet import MultiLeNetR
# from .mixed_curvature_layers import MixedCurvatureBlock


# class MixedCurvatureLeNetR(nn.Module):
#     """LeNet encoder with Mixed-Curvature post-processing.
    
#     Keeps the original LeNet conv feature extraction and adds mixed-curvature
#     embedding + encoder layers after the 50-dim output.
    
#     Args:
#         in_channels: input image channels (1 for MNIST)
#         embed_dim: embedding dimension (default 50, matching LeNet)
#         num_subspaces: number of curvature subspaces P (embed_dim must be divisible by P)
#         num_attn_heads: number of attention heads in the encoder layer
#     """

#     def __init__(self, in_channels=1, embed_dim=50, num_subspaces=5, num_attn_heads=5):
#         super().__init__()
#         assert embed_dim % num_subspaces == 0, (
#             f"embed_dim ({embed_dim}) must be divisible by num_subspaces ({num_subspaces})"
#         )
#         assert embed_dim % num_attn_heads == 0, (
#             f"embed_dim ({embed_dim}) must be divisible by num_attn_heads ({num_attn_heads})"
#         )

#         self.embed_dim = embed_dim

#         # Layer 1: Feature extraction (existing LeNet)
#         self.lenet = MultiLeNetR(in_channels=in_channels)

#         # Layer 2: Mixed-Curvature Embedding Layer
#         self.mc_embedding = MixedCurvatureBlock(dim=embed_dim, num_subspaces=num_subspaces)

#         # Layer 3: Mixed-Curvature Encoder Layer
#         self.mc_encoder = MixedCurvatureBlock(dim=embed_dim, num_subspaces=num_subspaces)

#         # Self-Attention (part of Encoder Layer)
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_attn_heads,
#             batch_first=True,
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)

#         # Feed-Forward Network
#         self.ffn_linear1 = nn.Linear(embed_dim, embed_dim * 2)
#         self.ffn_linear2 = nn.Linear(embed_dim * 2, embed_dim)

#         # Mark mixed-curvature and attention modules to NOT be lorafied
#         # PaLoRA's lorafy_model() would break these by converting their internal
#         # nn.Linear layers to PaLinear (which requires ray arguments).
#         # The LeNet conv/fc layers inside self.lenet WILL still be lorafied.
#         self.mc_embedding._skip_lorafy = True
#         self.mc_encoder._skip_lorafy = True
#         self.self_attn._skip_lorafy = True
#         self.ffn_linear1._skip_lorafy = True
#         self.ffn_linear2._skip_lorafy = True

#     def forward(self, x, ray: torch.Tensor = None):
#         """
#         Args:
#             x: (B, C, H, W) input image
#             ray: optional ray vector for PaLoRA compatibility
#         Returns:
#             (B, embed_dim) embedding
#         """
#         ray_kwargs = dict(ray=ray) if ray is not None else {}

#         # Step 1: LeNet feature extraction (lorafied layers get ray)
#         z = self.lenet(x, **ray_kwargs)

#         # Step 2: Mixed-Curvature Embedding Layer (NOT lorafied, no ray needed)
#         z = self.mc_embedding(z)

#         # Step 3: Mixed-Curvature Encoder Layer (NOT lorafied)
#         z = self.mc_encoder(z)

#         # Self-Attention (NOT lorafied)
#         z_seq = z.unsqueeze(1)  # (B, 1, D)
#         attn_out, _ = self.self_attn(z_seq, z_seq, z_seq)
#         z_seq = self.norm1(z_seq + attn_out)

#         # Feed-Forward + residual (NOT lorafied)
#         ffn_out = F.relu(self.ffn_linear1(z_seq))
#         ffn_out = self.ffn_linear2(ffn_out)
#         z_seq = self.norm2(z_seq + ffn_out)

#         return z_seq.squeeze(1)  # (B, D)

#     def get_last_layer(self):
#         """Return last layer for MGDA compatibility."""
#         return self.ffn_linear2


"""
MixedCurvatureLeNetR: LeNet encoder enhanced with Mixed-Curvature processing.

Architecture (following ICML 2025 paper's 3-layer design):
    1. LeNet feature extraction (existing MultiLeNetR) → 50-dim embedding
    2. Mixed-Curvature Embedding Layer: Exp → StereographicLinear → Log
    3. Mixed-Curvature Encoder Layer: Exp → StereographicLinear → Log → Self-Attention → FFN
    Output: 50-dim embedding (compatible with existing MultiLeNetO decoders)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lenet import MultiLeNetR
from .mixed_curvature_layers import MixedCurvatureBlock


class MixedCurvatureLeNetR(nn.Module):
    """LeNet encoder with Mixed-Curvature post-processing.
    
    Keeps the original LeNet conv feature extraction and adds mixed-curvature
    embedding + encoder layers after the 50-dim output.
    
    Args:
        in_channels: input image channels (1 for MNIST)
        embed_dim: embedding dimension (default 50, matching LeNet)
        num_subspaces: number of curvature subspaces P (embed_dim must be divisible by P)
    """

    def __init__(self, in_channels=1, embed_dim=50, num_subspaces=5):
        super().__init__()
        assert embed_dim % num_subspaces == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_subspaces ({num_subspaces})"
        )

        self.embed_dim = embed_dim

        # Layer 1: Feature extraction (existing LeNet)
        self.lenet = MultiLeNetR(in_channels=in_channels)

        # Layer 2: Mixed-Curvature Embedding Layer
        self.mc_embedding = MixedCurvatureBlock(dim=embed_dim, num_subspaces=num_subspaces)

        # Layer 3: Mixed-Curvature Encoder Layer
        self.mc_encoder = MixedCurvatureBlock(dim=embed_dim, num_subspaces=num_subspaces)

        # Mark mixed-curvature modules to NOT be lorafied
        self.mc_embedding._skip_lorafy = True
        self.mc_encoder._skip_lorafy = True

    def forward(self, x, ray: torch.Tensor = None):
        ray_kwargs = dict(ray=ray) if ray is not None else {}

        # Step 1: LeNet feature extraction (lorafied layers get ray)
        z = self.lenet(x, **ray_kwargs)

        # Step 2: Mixed-Curvature Embedding Layer
        z = self.mc_embedding(z)

        # Step 3: Mixed-Curvature Encoder Layer
        z = self.mc_encoder(z)

        return z

    def get_last_layer(self):
        """Return last layer for MGDA compatibility."""
        return self.mc_encoder.stereo_linear


