import torch

from .constants import DIV_THRESHOLD, X_LIM, Y_LIM


class MandelbrotDataloader:
    def __init__(
        self,
        mandelbrot_iterations: int,
        batch_size: int,
        epoch_len: int,
        device: str | torch.device,
        seed: int,
    ):
        self.mandelbrot_iterations = mandelbrot_iterations
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.device = device

        self.current_iter = 0
        self.rng = torch.Generator(self.device).manual_seed(seed)

    def generate_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        coords = torch.rand(
            (2, self.batch_size),
            generator=self.rng,
            dtype=torch.float,
            device=self.device,
        )
        coords[0] = X_LIM[0] + (X_LIM[1] - X_LIM[0]) * coords[0]
        coords[1] = Y_LIM[0] + (Y_LIM[1] - Y_LIM[0]) * coords[1]
        coords = torch.complex(real=coords[0], imag=coords[1])

        series = torch.zeros_like(coords, dtype=coords.dtype, device=coords.device)
        for _ in range(self.mandelbrot_iterations):
            series = series.pow(2) + coords

        return coords, series.abs() > DIV_THRESHOLD

    def __iter__(self) -> "MandelbrotDataloader":
        self.current_iter = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_iter > self.epoch_len:
            raise StopIteration

        self.current_iter += 1
        return self.generate_batch()
