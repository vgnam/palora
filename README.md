# PaLoRA

Official implementation of ["Pareto Low-Rank Adapters: Efficient Multi-Task Learning with Preferences" (ICLR 2025)](https://openreview.net/forum?id=icDoYdUhRa).

## Overview

This repository contains the official implementation of PaLoRA (Pareto Low-Rank Adapters), a novel approach for efficient multi-task learning with preferences. Our method enables effective multi-task optimization while considering task preferences and maintaining computational efficiency through low-rank adaptations.

The paper is available at: [OpenReview](https://openreview.net/forum?id=icDoYdUhRa)


## Running experiments

All experiments were run with NVIDIA GPUs, predominantly V100-SXM2-32GB. No experiment requires more than one GPU.
All the experiments are in the same directory as `src/` and are defined in separate files. 
If the user does not modify the configuration, PaLoRA is used by default.
To run one experiment, you simply:

```bash
# MultiMNIST experiment with PaLoRA
python _multimnist.py

# ...or with NashMTL
python _multimnist.py method=nashmtl

# NYU experiment with PaLoRA and changing the batch size
# and disabling Weights and Biases
python _nyuv2.py data.batch_size=16 wandb.mode=disabled

# CityScapes experiment specifying the rank of the low-rank adaptors
python _split-cityscapes.py method.rank=8
```




We use [hydra](https://hydra.cc/) for configuration management, all configs can be found in `configs/` and they are structured as follows:

```txt
configs
├── experiment
│   ├── cityscapes
│   │   ├── cityscapes.yaml
│   │   └── method
│   │       ├── cosmos.yaml
│   │       ├── full-pamal.yaml
│   │       ├── palora.yaml
│   │       ├── pamal.yaml
│   │       └── phn.yaml
│   ├── multimnist
│   │   ├── debug.yaml
│   │   ├── method
│   │   │   ├── cosmos.yaml
│   │   │   ├── palora.yaml
│   │   │   ├── pamal.yaml
│   │   │   └── phn.yaml
│   │   └── multimnist.yaml
│   ├── multimnist3
│   │   ├── method
│   │   │   ├── cosmos.yaml
│   │   │   ├── palora.yaml
│   │   │   ├── pamal.yaml
│   │   │   └── phn.yaml
│   │   └── multimnist3.yaml
│   ├── nyuv2
│   │   ├── method
│   │   │   ├── cosmos.yaml
│   │   │   ├── full-pamal.yaml
│   │   │   ├── palora.yaml
│   │   │   ├── pamal.yaml
│   │   │   └── phn.yaml
│   │   └── nyuv2.yaml
│   ├── resume.yaml
│   └── utkface
│       ├── method
│       │   ├── cosmos.yaml
│       │   ├── palora.yaml
│       │   ├── pamal.yaml
│       │   └── phn.yaml
│       └── utkface.yaml
└── general
    ├── data
    │   ├── cityscapes.yaml
    │   ├── multimnist3.yaml
    │   ├── multimnist.yaml
    │   ├── nyuv2.yaml
    │   └── utkface.yaml
    ├── hydra_cfg.yaml
    ├── method
    │   ├── autol.yaml
    │   ├── cagrad.yaml
    │   ├── dwa.yaml
    │   ├── graddrop.yaml
    │   ├── imtl.yaml
    │   ├── ls.yaml
    │   ├── mgda.yaml
    │   ├── nashmtl.yaml
    │   ├── pcgrad.yaml
    │   ├── rlw.yaml
    │   ├── stl0.yaml
    │   ├── stl1.yaml
    │   ├── .... more methods
    │   └── uw.yaml
    ├── model
    │   ├── lenet.yaml
    │   ├── psp.yaml
    │   └── unet.yaml
    ├── optimizer
    │   ├── adam_defaults.yaml
    │   └── sgd_defaults.yaml
    ├── ray_sampler
    │   ├── annealing_dirichlet.yaml
    │   ├── annealing.yaml
    │   ├── dirichlet.yaml
    │   └── fixed.yaml
    ├── sampling
    │   └── constant.yaml
    ├── scheduler
    │   ├── cyclic.yaml
    │   └── multistep.yaml
    └── wandb
        └── wandb.yaml
```




## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{
  dimitriadis2025pareto,
  title={Pareto Low-Rank Adapters: Efficient Multi-Task Learning with Preferences},
  author={Nikolaos Dimitriadis and Pascal Frossard and Fran{\c{c}}ois Fleuret},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=icDoYdUhRa}
}
```