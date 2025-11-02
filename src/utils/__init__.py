from .logging_utils import install_logging, initialize_wandb
import numpy as np
import torch
import random
import logging


def set_seed(seed=-1):
    """for reproducibility
    :param seed:
    :return:
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    if seed != -1:
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    logging.info(f"Setting seed to {torch.initial_seed()}")


def safe_pop(config, key="name"):
    if getattr(config, key, None) is not None:
        config = dict(config)
        config.pop(key)
        return config
    return config


__all__ = ["set_seed", "safe_pop", "install_logging", "initialize_wandb"]
