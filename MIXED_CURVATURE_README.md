# Mixed-Curvature PaLoRA

Extension of [PaLoRA](https://arxiv.org/abs/2402.xxxxx) (ICLR 2025) with a **Mixed-Curvature encoder** inspired by *"A Mixed-Curvature based Pre-training Paradigm for Multi-Task Vehicle Routing Solver"* (ICML 2025).

## Architecture

The encoder adds mixed-curvature processing **after** the standard LeNet feature extraction:

```
Input (1×36×36)
  │
  ▼
┌─────────────────────────────────────┐
│  LeNet Feature Extraction           │  ← Conv → Conv → FC (existing)
│  Conv2d(1→10) → MaxPool → ReLU     │
│  Conv2d(10→20) → MaxPool → ReLU    │
│  Linear(320→50) → ReLU             │
└──────────────┬──────────────────────┘
               │ 50-dim embedding
               ▼
┌─────────────────────────────────────┐
│  Mixed-Curvature Embedding Layer    │  ← NEW
│                                     │
│  z_manifold = Exp₀(z, κ_list)       │  Project to P manifold subspaces
│  z_curved = StereographicLinear(z)  │  Möbius matvec in curved space
│  z_out = Log₀(z_curved, κ_list)    │  Back to tangent space
│  output = a·z + b·z_out            │  Learnable residual
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Mixed-Curvature Encoder Layer      │  ← NEW
│                                     │
│  Same Exp → Linear → Log → Residual│
│  + Multi-Head Self-Attention        │
│  + Feed-Forward Network             │
│  + Layer Normalization              │
└──────────────┬──────────────────────┘
               │ 50-dim embedding
               ▼
┌─────────────────────────────────────┐
│  Task Decoders (per task)           │  ← existing MultiLeNetO
│  Linear(50→50) → ReLU              │
│  Linear(50→10) → LogSoftmax        │
└─────────────────────────────────────┘
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `StereographicModel` | `src/models/factory/manifolds.py` | Wrapper for geoopt stereographic math (Exp/Log maps, Möbius ops) |
| `StereographicLinear` | `src/models/factory/mixed_curvature_layers.py` | Linear layer via Möbius matvec with learnable curvatures per subspace |
| `MixedCurvatureBlock` | `src/models/factory/mixed_curvature_layers.py` | Exp → StereographicLinear → Log → residual |
| `MixedCurvatureLeNetR` | `src/models/factory/mixed_curvature_lenet.py` | Full encoder: LeNet + MC Embedding + MC Encoder + Attention |

### Mixed-Curvature Subspaces

The 50-dim embedding is split into **P=5 subspaces** of 10 dims each. Each subspace has a **learnable curvature κₚ**:

| κ < 0 | κ = 0 | κ > 0 |
|-------|-------|-------|
| Hyperbolic (Poincaré ball) | Euclidean (flat) | Spherical |
| Captures hierarchical structure | Standard similarity | Angular/directional relationships |

Curvatures are initialized to **0** (Euclidean) and learned during training — the model discovers the optimal geometry.

## Usage

### Run with Mixed-Curvature encoder

```bash
python _multimnist.py model.encoder=MixedCurvatureLeNetR training.epochs=100
```

### Run with original LeNet encoder (baseline)

```bash
python _multimnist.py training.epochs=100
```

### PaLoRA Integration

- PaLoRA's `lorafy_model()` applies LoRA adapters to the **LeNet layers** (Conv2d → PaConv2d, Linear → PaLinear)
- Mixed-curvature modules are **excluded** from lorafication via `_skip_lorafy` flag
- The mixed-curvature layers have their own learnable parameters (curvatures κ, StereographicLinear weights)

## Dependencies

Requires `geoopt >= 0.4.0` (added to `requirements.txt`):

```bash
pip install geoopt
```

## References

- **PaLoRA**: Pareto Low-Rank Adapters (ICLR 2025)
- **Mixed-Curvature VRPs**: *A Mixed-Curvature based Pre-training Paradigm for Multi-Task Vehicle Routing Solver* (ICML 2025, [GitHub](https://github.com/lsyysl9711/Mixed_Curvature_VRPs))
- **geoopt**: Riemannian Optimization in PyTorch ([GitHub](https://github.com/geoopt/geoopt))
