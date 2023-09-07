from functools import partial
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
    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()

        Linear = partial(nn.Linear, dtype=torch.cfloat)
        LazyLinear = partial(nn.LazyLinear, dtype=torch.cfloat)

        self.project_input = nn.Sequential(
            LazyLinear(hidden_dim),
            CReLU(),
            CLayerNorm(hidden_dim),
        )

        self.residual_layers = nn.ModuleList(
            [
                nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    CReLU(),
                    CLayerNorm(hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )

        self.project_output = Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_input(x)

        for layer in self.residual_layers:
            x = x + layer(x)

        x = self.project_output(x)

        x = x.angle()  # Real value in range [-pi, pi].
        return torch.sigmoid(x)  # Real value in range [0.04, 0.96].
