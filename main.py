import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from src.dataset import NNandelbrotDataloader
from src.model import NNandelbrotModel
from src.trainer import NNandelbrotTrainer


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig):
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NNandelbrotModel(
        config.model.hidden_dim,
        config.model.n_layers,
    )
    dataloader = NNandelbrotDataloader(
        config.data.mandelbrot_iterations,
        config.train.batch_size,
        config.train.epoch_len,
        config.device,
        config.seed,
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr)
    trainer = NNandelbrotTrainer(model, optimizer, dataloader, config.device)

    trainer.train(config.group, OmegaConf.to_container(config), config.mode)


if __name__ == "__main__":
    # Launch with hydra.
    main()
