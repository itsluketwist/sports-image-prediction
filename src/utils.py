from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List

import torch


class ModelOptions(Enum):
    SPORTS = "sports"
    KMNIST = "kmnist"

    def __str__(self):
        return self.value


def filename_datetime() -> str:
    """
    Generate a filename friendly datetime string.

    Returns
    -------
    str
        The filename string.
    """
    return datetime.now().strftime("%Y_%m_%d_T%H%M")


def get_device() -> torch.device:
    """
    Get the best available device for training.

    Returns
    -------
    torch.device
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)

    def update(
        self,
        train_loss: float,
        train_accuracy: float,
        test_loss: float,
        test_accuracy: float,
    ):
        """
        Update the training class with results from a training loop.

        Parameters
        ----------
        train_loss: float
        train_accuracy: float
        test_loss: float
        test_accuracy: float
        """
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.test_loss.append(test_loss)
        self.test_accuracy.append(test_accuracy)
