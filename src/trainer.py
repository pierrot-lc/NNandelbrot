from typing import Any

import torch
import torch.optim as optim
from tqdm import tqdm

import wandb

from .dataset import NNandelbrotDataloader
from .generate import generate
from .model import NNandelbrotModel


class NNandelbrotTrainer:
    def __init__(
        self,
        model: NNandelbrotModel,
        optimizer: optim.Optimizer,
        dataloader: NNandelbrotDataloader,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader

        self.loss = torch.nn.BCELoss()

    def do_batch(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        metrics = dict()

        y_pred = self.model(x)
        metrics["loss"] = self.loss(y_pred, y)

        return metrics

    def train(self, group: str, config: dict[str, Any], mode: str):
        with wandb.init(
            project="NNandelbrot",
            entity="pierrotlc",
            group=group,
            config=config,
            mode=mode,
        ) as run:
            for _ in tqdm(range(10), desc="Epochs", disable=mode == "disabled"):
                for x, y in tqdm(
                    self.dataloader,
                    desc="Batchs",
                    leave=False,
                    disable=mode == "disabled",
                ):
                    metrics = self.do_batch(x, y)
                    metrics["loss"].backward()
                    self.optimizer.step()

                metrics = self.evaluate()
                run.log(metrics)

    def evaluate(self) -> dict[str, Any]:
        metrics = dict()
        image = generate(self.model, width=600, height=400)
        metrics["image"] = wandb.Image(image)
        return metrics
