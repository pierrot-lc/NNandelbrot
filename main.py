from src.dataset import NNandelbrotDataloader
from src.model import NNandelbrotModel
from src.trainer import NNandelbrotTrainer

import torch
import torch.optim as optim


def main():
    model = NNandelbrotModel(1)
    dataloader = NNandelbrotDataloader(100, 128, 100, "cpu", 0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    trainer = NNandelbrotTrainer(model, optimizer, dataloader)
    trainer.train()


if __name__ == "__main__":
    main()
