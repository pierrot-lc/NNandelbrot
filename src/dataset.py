import torch

from .constants import DIV_THRESHOLD, X_LIM, Y_LIM


class NNandelbrotDataloader:
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

    def sample_uniform(self) -> torch.Tensor:
        """Generate a batch of complex points.
        They are sampled uniformly over the mandelbrot space.

        ---
        Returns:
            The sampled points.
                Complex tensor of shape [batch_size,].
        """
        coords = torch.rand(
            (2, self.batch_size),
            generator=self.rng,
            dtype=torch.float,
            device=self.device,
        )
        coords[0] = X_LIM[0] + (X_LIM[1] - X_LIM[0]) * coords[0]
        coords[1] = Y_LIM[0] + (Y_LIM[1] - Y_LIM[0]) * coords[1]
        return torch.complex(real=coords[0], imag=coords[1])

    def mandelbrot_labels(self, points: torch.Tensor) -> torch.Tensor:
        """Apply the original Mandelbrot iterations to return the binary label
        of the given points.

        ---
        Args:
            points: Points for which we calculate the labels.
                Complex tensor of shape [batch_size,].

        ---
        Returns:
            The corresponding labels.
                Boolean tensor of shape [batch_size,].
        """
        series = torch.zeros_like(points, dtype=points.dtype, device=points.device)

        for _ in range(self.mandelbrot_iterations):
            series = series.pow(2) + points

        return series.abs() > DIV_THRESHOLD

    def __len__(self) -> int:
        return self.epoch_len

    def __iter__(self) -> "NNandelbrotDataloader":
        self.current_iter = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_iter > self.epoch_len:
            raise StopIteration

        self.current_iter += 1

        points = self.sample_uniform()
        labels = self.mandelbrot_labels(points)
        return points.unsqueeze(-1), labels.float().unsqueeze(-1)
