import torch
import torch.optim as optim

from .dataset import NNandelbrotDataloader
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

    def train(self):
        for x, y in self.dataloader:
            metrics = self.do_batch(x, y)
            metrics["loss"].backward()
            self.optimizer.step()

            print(metrics["loss"])
