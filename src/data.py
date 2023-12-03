from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from typing import Tuple
import torch


def get_kmnist_dataloaders(
	train_split: float, batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
	print("Loading the KMNIST dataset...")
	train_data = KMNIST(root="data", train=True, download=True, transform=ToTensor(),)
	test_data = KMNIST(root="data", train=False, download=True, transform=ToTensor(),)

	print("Generating the data sets...")
	num_train = int(len(train_data) * train_split)
	num_val = int(len(train_data) * (1 - train_split))

	(train_data, valData) = random_split(
		dataset=train_data,
		lengths=[num_train, num_val],
		generator=torch.Generator().manual_seed(42),
	)

	train_dl = DataLoader(train_data, shuffle=True, batch_size=batch_size,)
	val_dl = DataLoader(valData, batch_size=batch_size,)
	test_dl = DataLoader(test_data, batch_size=batch_size,)

	return (train_dl, val_dl, test_dl, len(train_data.dataset.classes))
