import logging
from inspect import signature

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb
from src.callbacks import get_default_callbacks
from src.callbacks.methods import METHODS, PFL_METHODS
from src.callbacks.methods.algo_callback import ParetoFrontApproximationAlgoCallback
from src.callbacks.methods.palora import PaLoRA
from src.callbacks.methods.pamal import PaMaL
from src.callbacks.metrics.mtl_metric_callback import (
    ClassificationMultiTaskMetricCallback,
)
from src.callbacks.utils.cosine_similarity_callback import CosineSimilarityCallback
from src.callbacks.utils.pareto_front_visualizer import ParetoFrontVisualizerCallback
from src.callbacks.utils.save_model import SaveModelCallback
from src.datasets import MultiMnistDataModule
from src.models.base_model import SharedBottom
from src.models.factory.lenet import MultiLeNetO, MultiLeNetR
from src.models.factory.mixed_curvature_lenet import MixedCurvatureLeNetR
from src.trainer.base_trainer import BaseTrainer
from src.trainer.ensemble_trainer import EnsembleTrainer
from src.trainer.multi_forward_ensemble_trainer import MultiForwardEnsembleTrainer
from src.utils import initialize_wandb, install_logging, safe_pop, set_seed
from src.utils.losses import MultiTaskCrossEntropyLoss


@hydra.main(config_path="configs/experiment/multimnist", config_name="multimnist")
def my_app(config: DictConfig) -> None:
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    set_seed(config.seed)

    initialize_wandb(config)

    wandb.run.tags = ("ablation",)

    dm: MultiMnistDataModule = instantiate(safe_pop(config.data, key="dataset"))
    logging.info(f"I am using the following benchmark {dm.name}")

    if config.method.name in PFL_METHODS:
        ray_sampler = instantiate(safe_pop(config.ray_sampler), num=config.method.num, num_tasks=dm.num_tasks)

        weight_method = METHODS[config.method.name](
            **safe_pop(config.method), ray_sampler=ray_sampler, num_tasks=dm.num_tasks
        )

    else:
        weight_method = METHODS[config.method.name](dm.num_tasks, **safe_pop(config.method))
    logging.info(f"I am using the following method {weight_method}")

    if getattr(config.model, 'encoder', 'MultiLeNetR') == 'MixedCurvatureLeNetR':
        encoder = MixedCurvatureLeNetR(in_channels=1)
    else:
        encoder = MultiLeNetR(in_channels=1)

    model = SharedBottom(
        encoder=encoder,
        decoder=MultiLeNetO(),
        task_names=dm.task_names,
    )
    model = weight_method.configure_model(model, data_module=dm)

    param_groups = weight_method.configure_param_groups(model, lr=config.optimizer.lr)

    optimizer = torch.optim.Adam(param_groups, lr=config.optimizer.lr)

    metric_cb = ClassificationMultiTaskMetricCallback(use_task_names=True)
    callbacks = [
        metric_cb,
        *get_default_callbacks(),
        SaveModelCallback(epoch_frequency=2),
        ParetoFrontVisualizerCallback(),
        CosineSimilarityCallback(),
    ]

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        gpu=0,
        callbacks=callbacks,
        loss_fn=MultiTaskCrossEntropyLoss(),
    )

    if isinstance(weight_method, ParetoFrontApproximationAlgoCallback):
        trainer_kwargs["validate_every_n"] = config.validate_every_n
        if getattr(weight_method, "num", 1) > 1:
            trainer = MultiForwardEnsembleTrainer(method=weight_method, **trainer_kwargs)
        else:
            trainer = EnsembleTrainer(method=weight_method, **trainer_kwargs)
    else:
        trainer = BaseTrainer(method=weight_method, **trainer_kwargs)
    print(trainer)
    trainer.fit(epochs=config.training.epochs)
    trainer.predict(dm.test_dataloader())

    wandb.finish(quiet=True)


if __name__ == "__main__":
    my_app()
