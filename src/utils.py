# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Utility functions for the NMT project.

This module provides generic helper functions used throughout the project, such as
setting random seeds for reproducibility, counting trainable parameters,
saving/loading PyTorch checkpoints, and plotting training metrics.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import OUTPUT_DIR


def set_seed(seed: int = 42) -> None:
    """Sets the random seed across all libraries to ensure reproducible results.

    This locks the random number generators for standard Python `random`, NumPy,
    and PyTorch (CPU, CUDA, and MPS backends).

    Args:
        seed (int, optional): The integer seed to lock randomness. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic cuDNN algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Calculates the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The instantiated PyTorch model.

    Returns:
        int: The total count of parameters that have `requires_grad=True`.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_loss_curve(train_losses: List[float], val_losses: List[float], path: Path) -> None:
    """Generates and saves a matplotlib plot of training and validation loss over epochs.

    Args:
        train_losses (List[float]): A list of training loss values per epoch.
        val_losses (List[float]): A list of validation loss values per epoch.
        path (Path): The destination path where the PNG image will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, path: Path
) -> None:
    """Saves the complete state of the model and optimizer to disk.

    Saving the optimizer state is crucial to accurately resume training later.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The validation loss at this checkpoint.
        path (Path): The destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module, path: Path, optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[int, float]:
    """Loads a previously saved model and optimizer state from disk.

    Forces the tensors to be loaded onto the CPU first to avoid VRAM spikes,
    which is standard practice in PyTorch.

    Args:
        model (torch.nn.Module): The uninitialized model structure.
        path (Path): The file path of the `.pt` checkpoint.
        optimizer (Optional[torch.optim.Optimizer], optional): The optimizer. Defaults to None.

    Raises:
        FileNotFoundError: If the specified checkpoint path does not exist.

    Returns:
        Tuple[int, float]: The epoch and the recorded validation loss from the checkpoint.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    # Load to CPU first to prevent device-mismatch issues.
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


def ensure_output_dirs() -> None:
    """Creates necessary project directories if they do not currently exist.

    Ensures that scripts will not fail with `FileNotFoundError` when attempting
    to write outputs to `OUTPUT_DIR`.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def qualitative_comment(_src: str, _ref: str, _hyp: str) -> str:
    """Generates a qualitative comment based on the translation comparison.

    Args:
        _src (str): The source sentence.
        _ref (str): The true reference target sentence.
        _hyp (str): The hypothesized generated translation.

    Returns:
        str: A placeholder comment string for analysis.
    """
    return "Qualitative analysis to be added."
