from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import KMNIST, ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


SPORTS_IMAGE_SIZE = (224, 224)
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


sports_transform = Compose(
    [
        ToTensor(),
        Normalize(MEAN, STD),
        Resize(SPORTS_IMAGE_SIZE),
    ],
)


def get_kmnist_train_data(
    train_split: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for training a model with the KMNIST data set.

    Parameters
    ----------
    train_split: float
        What percentage of data should be used for training?
    batch_size: int
        The training batch size.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Tuple of training data and testing data.
    """
    train_data = KMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    num_train = int(len(train_data) * train_split)
    num_test = int(len(train_data) * (1 - train_split))

    (train_data, test_data) = random_split(
        dataset=train_data,
        lengths=[num_train, num_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
    )
    return (train_loader, test_loader)


def get_kmnist_eval_data() -> DataLoader:
    """
    Create a dataloader for evaluating a model with the KMNIST data set.

    Returns
    -------
    DataLoader
        Dataloader of evaluation data.
    """
    eval_data = KMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    eval_loader = DataLoader(
        eval_data,
        batch_size=len(eval_data),
    )

    return eval_loader


def get_kmnist_sample(
    count: int,
) -> DataLoader:
    """
    Create a dataloader of KMNIST data samples.

    Parameters
    ----------
    count: int
        How many samples to load.

    Returns
    -------
    DataLoader
        Dataloader of KMNIST samples.
    """
    data = KMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    idxs = np.random.choice(range(0, len(data)), size=(count,))
    _subset = Subset(data, idxs)
    return DataLoader(_subset, batch_size=1)


def _sports_image_folder(type: str) -> ImageFolder:
    """
    Create an ImageFolder instance for loading sports data.

    Parameters
    ----------
    type: str
        Which sports image data type/directory to load.

    Returns
    -------
    ImageFolder
    """
    return ImageFolder(
        f"data/sports/{type}",
        transform=sports_transform,
    )


def get_sports_train_data(
    batch_size: int,
    use_percent: float = 1.0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for training a model with the sports image data set.

    Parameters
    ----------
    batch_size: int
        The training batch size.
    use_percent: float = 1.0
        What percentage of data to use, reducing the amount can
        speed up training for initial model validation.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Tuple of training data and testing data.
    """
    train_data = _sports_image_folder("train")
    test_data = _sports_image_folder("test")

    if use_percent != 1:
        # reduce size of datasets to speed up training/testing
        train_data = Subset(
            train_data,
            np.random.choice(
                len(train_data), int(use_percent * len(train_data)), replace=False
            ),
        )
        test_data = Subset(
            test_data,
            np.random.choice(
                len(test_data), int(use_percent * len(test_data)), replace=False
            ),
        )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
    )

    return (train_loader, test_loader)


def get_sports_eval_data() -> DataLoader:
    """
    Create a dataloader for evaluating a model with the sports image data set.

    Returns
    -------
    DataLoader
       Dataloader of evaluation data.
    """
    eval_data = _sports_image_folder("eval")
    eval_loader = DataLoader(
        eval_data,
        batch_size=len(eval_data),
    )

    return eval_loader


def get_sports_sample(
    count: int,
) -> DataLoader:
    """
    Create a dataloader of sport image data samples.

    Parameters
    ----------
    count: int
        How many samples to load.

    Returns
    -------
    DataLoader
        Dataloader of sport image samples.
    """
    data = _sports_image_folder("eval")
    idxs = np.random.choice(range(0, len(data)), size=(count,))
    _subset = Subset(data, idxs)
    return DataLoader(_subset, batch_size=1)


def load_sports_image(path: str) -> tuple[torch.Tensor, str]:
    """
    Load a single sports image from file, into a tensore.
    Filename must contain the image class label.

    Parameters
    ----------
    path: str

    Returns
    -------
    tuple[torch.Tensor, str]
        Tuple of the image as a tensore, and the string label.
    """
    label = Path(path).stem.lower()

    with open(path, "rb") as file:
        image = Image.open(file)
        image_tensor = sports_transform(img=image)
        image_tensor = image_tensor.unsqueeze_(0)

    return (image_tensor, label)
