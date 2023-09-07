from typing import Any

import torch
import torch.nn as nn
from torch.nn.modules.normalization import _shape_t


class CReLU(nn.ReLU):
    """Implementation of the complex ReLU.
    It simply applies the ReLU separately on the real and imaginary part.
    See: https://arxiv.org/abs/1705.09792.
    """

    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_real(x)
        x = super().forward(x)
        x = torch.view_as_complex(x)
        return x


class CLayerNorm(nn.LayerNorm):
    """My easy implementation of what could be a complex LayerNorm."""

    def __init__(
        self, normalized_shape: _shape_t, *args: list[Any], **kwargs: dict[str, Any]
    ):
        super().__init__(normalized_shape, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angle = x.angle()
        magnitude = x.abs()

        normalized_magnitude = super().forward(magnitude)
        normalized_x = normalized_magnitude * torch.exp(1j * angle)
        return normalized_x


class NNandelbrotModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        dtype = torch.cfloat

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 5, dtype=dtype),
            CReLU(),
            CLayerNorm(5),
            nn.Linear(5, 5, dtype=dtype),
            CReLU(),
            CLayerNorm(5),
            nn.Linear(5, 5, dtype=dtype),
            CReLU(),
            CLayerNorm(5),
            nn.Linear(5, 1, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = x.angle()  # Real value in range [-pi, pi].
        return torch.sigmoid(x)  # Real value in range [0.04, 0.96].
