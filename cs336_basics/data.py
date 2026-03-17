import numpy as np
import numpy.typing as npt
import torch
import os
from typing import IO, BinaryIO


def data_load(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    if max_start <= 0:
        raise ValueError("dataset must be longer than context_length")

    start_indices = np.random.randint(0, max_start, size=batch_size)
    inputs = np.stack([dataset[start: start + context_length]
                      for start in start_indices])
    targets = np.stack([dataset[start + 1: start + context_length + 1]
                       for start in start_indices])

    input_tensor = torch.as_tensor(inputs, dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
    return input_tensor, target_tensor


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {"model_state": model_state,
                  "optim_state": optim_state, "iteration": iteration}
    torch.save(checkpoint, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model_state, optim_state, iteration = checkpoint[
        "model_state"], checkpoint["optim_state"], checkpoint["iteration"]
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)
    return iteration
