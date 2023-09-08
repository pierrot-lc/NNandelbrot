import torch

from .constants import X_LIM, Y_LIM
from .model import NNandelbrotModel


def generate(
    model: NNandelbrotModel,
    width: int,
    height: int,
    x_range: tuple[int, int] = X_LIM,
    y_range: tuple[int, int] = Y_LIM,
) -> torch.Tensor:
    x = torch.linspace(x_range[0], x_range[1], width)
    y = torch.linspace(y_range[0], y_range[1], height)
    points = torch.cartesian_prod(y, x)  # Shape of [height x width, 2].

    points = torch.view_as_complex(points)
    y_pred = model(points.unsqueeze(-1))
    labels = y_pred > 0.5

    labels = labels.reshape(height, width).float()
    return labels
