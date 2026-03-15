# PaMaL-MC-Div

**PaMaL + Mixed-Curvature + Diversity Regularization**

Method mới kết hợp PaMaL subspace learning với Mixed-Curvature geometry và magnitude diversity regularization.

## Kiến trúc

```
Input (1×36×36)
  │
  ▼
┌───────────────────────────────────────┐
│  LeNet Encoder (subspace by PaMaL)    │  ← Conv→Conv→FC, mỗi layer có N copies
│  SubspaceConv, SubspaceLinear         │     trộn bởi ray weights: w = Σ αᵢ·wᵢ
└──────────────┬────────────────────────┘
               │ 50-dim embedding
               ▼
┌───────────────────────────────────────┐
│  MC Encoder Block                     │  ← NEW (không bị subspace-ify)
│  Exp₀(z, κ) → StereographicLinear    │     Learnable curvatures κ per subspace
│  → Log₀ → a·z + b·z_curved           │     P=5 subspaces × 10 dims each
└──────────────┬────────────────────────┘
               │
               ├──── ★ L_diverse = -Magnitude(z_mc across rays)
               │         Magnitude = effective distinct points (magnipy-inspired)
               ▼
┌───────────────────────────────────────┐
│  MC Decoder Block                     │  ← NEW (không bị subspace-ify)
│  Same Exp→Linear→Log→Residual        │
└──────────────┬────────────────────────┘
               │ 50-dim
               ▼
┌───────────────────────────────────────┐
│  Task Decoders (subspace by PaMaL)    │  ← 1 per task
│  SubspaceLinear → SubspaceLinear      │
│  → LogSoftmax                         │
└───────────────────────────────────────┘
```

## So sánh với PaMaL gốc

| | **PaMaL** | **PaMaL-MC-Div** |
|---|---|---|
| Encoder | SubspaceConv/Linear | SubspaceConv/Linear + MC Block |
| Latent space | Euclidean only | Mixed-Curvature (κ<0: hyperbolic, κ=0: Euclidean, κ>0: spherical) |
| Diversity | Không có | Magnitude diversity trên MC embedding |
| Params thêm | — | MC blocks (~10K params), diversity coefficient |

## Files

| File | Mô tả |
|------|--------|
| `src/callbacks/methods/pamal_mc_div.py` | Method class `PaMaLMCDiv` |
| `src/models/factory/manifolds.py` | `StereographicModel` (geoopt wrapper) |
| `src/models/factory/mixed_curvature_layers.py` | `StereographicLinear`, `MixedCurvatureBlock` |
| `src/models/base_model.py` | `SharedBottom.forward()` — MC block integration |
| `configs/experiment/multimnist/method/pamal_mc_div.yaml` | Config |

## Sử dụng

```bash
# PaMaL-MC-Div (50 epochs, wandb disabled)
python _multimnist.py method=pamal_mc_div training.epochs=50 wandb.mode=disabled

# Baseline PaMaL
python _multimnist.py method=pamal training.epochs=50 wandb.mode=disabled
```

## Hyperparameters

| Parameter | Default | Mô tả |
|-----------|---------|--------|
| `diversity_coefficient` | 0.1 | Trọng số L_diverse trong total loss |
| `num_subspaces` | 5 | Số subspaces (P), mỗi cái có κ riêng |
| `embed_dim` | 50 | Chiều embedding (phải chia hết cho P) |
| `num` | 4 | Số rays per training step |
| `reg_coefficient` | 0 | PaMaL regularization (original) |

## Cơ chế hoạt động

1. **PaMaL subspace**: Mỗi layer có N copies weights, trộn theo ray `w = Σ αᵢ·wᵢ`
2. **MC blocks**: Project embedding qua multiple curvature spaces (Exp/Log maps)
3. **Magnitude-based diversity loss** (inspired by [magnipy](https://github.com/aidos-lab/magnipy)):
   - Cho mỗi sample trong batch, tính pairwise distance giữa MC embeddings của các rays
   - Xây dựng similarity matrix: `Z_ij = exp(-t · d_ij)`
   - Giải hệ `Z · w = 1` để tìm magnitude weights
   - Magnitude = `sum(w)` = "effective number of distinct points"
   - Loss = `-magnitude` → tối đa hóa magnitude = tối đa hóa diversity
   - Ma trận Z chỉ có kích thước `num_rays × num_rays` (4×4) → rất nhanh

## Dependencies

- `geoopt >= 0.4.0`
