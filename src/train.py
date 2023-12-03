from model import LeNet
from data import get_kmnist_dataloaders
from loop import training_loop, validation_loop, History, evaluation_loop
from utils import filename_datetime, get_device
from plot import plot_history

from torch.optim import Adam
from torch import nn
import torch
import time
import logging


logger = logging.getLogger(__name__)


# default training hyperparameters
DEFAULT_INIT_LEARN_RATE = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_TRAIN_SPLIT = 0.75


def run_train(
	init_learning_rate: float = DEFAULT_INIT_LEARN_RATE,
	batch_size: int = DEFAULT_BATCH_SIZE,
	epochs: int = DEFAULT_EPOCHS,
	train_split: float = DEFAULT_TRAIN_SPLIT,
	model_path: str = "output",
	plot_path: str = "output",
):
	device = get_device()  # configure the device to use

	# get the dataloaders
	train_dl, val_dl, test_dl, classes = get_kmnist_dataloaders(
		train_split=train_split, batch_size=batch_size,
	)

	logger.info("Initializing the model...")
	model = LeNet(num_channels=1, classes=classes,).to(device)
	opt = Adam(model.parameters(), lr=init_learning_rate)  # init optimizer
	lossFn = nn.NLLLoss()  # init loss function
	hist = History()  # init history dict

	print("Beginning to train the network...")
	for e in range(0, epochs):
		train_loss, train_accuracy = training_loop(
			model=model, loader=train_dl, loss_func=lossFn, optimizer=opt,
		)
		val_loss, val_accuracy = validation_loop(
			model=model, loader=val_dl, loss_func=lossFn,
		)
		hist.update(
			train_loss=train_loss.cpu().detach().numpy(),
			train_accuracy=train_accuracy,
			test_loss=val_loss.cpu().detach().numpy(),
			test_accuracy=val_accuracy,
		)

		# print the model training and validation information
		logger.info("EPOCH: {}/{}".format(e + 1, epochs))
		logger.info(f"Train loss: {train_loss:.6f}, Train accuracy: {train_accuracy:.4f}")
		logger.info(f"Val loss: {val_loss:.6f}, Val accuracy: {val_accuracy:.4f}\n")

	# we can now evaluate the network on the test set
	logger.info("Evaluating network...")
	evaluation_loop(
		model=model,
		loader=test_dl,
	)

	plot_history(
		history=hist,
		save_location=plot_path,
	)

	# save the model to disk
	torch.save(
		obj=model,
		f=f"{model_path.rstrip('/')}/model_{filename_datetime}.pth",
	)
