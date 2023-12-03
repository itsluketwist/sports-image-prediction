from dataclasses import dataclass, field
from typing import List
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import numpy as np


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
		self.train_loss.append(train_loss)
		self.train_accuracy.append(train_accuracy)
		self.test_loss.append(test_loss)
		self.test_accuracy.append(test_accuracy)


def training_loop(
	model: Module, loader: DataLoader, loss_func: _Loss, optimizer: Optimizer,
):
	model.train()  # model needs to be in evaluation mode
	steps = len(loader)
	data_size = len(loader.dataset)
	total_loss, num_correct = 0, 0

	# loop over the training set
	for (X, y) in loader:
		# send the input to the device
		# (x, y) = (x.to(device), y.to(device))
		# perform a forward pass and calculate the training loss

		pred = model(X)  # perform forward pass
		loss = loss_func(pred, y)  # calculate the loss

		optimizer.zero_grad()  # zero out the gradients
		loss.backward()  # perform backpropagation
		optimizer.step()  # update the weights

		total_loss += loss  # track loss
		num_correct += (
			(pred.argmax(1) == y).type(torch.float).sum().item()
		)  # track correctness

	# return the training statistics
	average_loss = total_loss / steps
	accuracy = num_correct / data_size
	return average_loss, accuracy


def validation_loop(
	model: Module, loader: DataLoader, loss_func: _Loss,
):
	model.eval()  # model needs to be in evaluation mode
	steps = len(loader)
	data_size = len(loader.dataset)
	total_loss, num_correct = 0, 0

	# switch off autograd for validation
	with torch.no_grad():

		for (X, y) in loader:
			# send the input to the device
			# (X, y) = (X.to(device), y.to(device))

			# make the predictions and calculate the validation loss
			pred = model(X)
			total_loss += loss_func(pred, y)  # track loss
			num_correct += (
				(pred.argmax(1) == y).type(torch.float).sum().item()
			)  # track correctness

	# return the training statistics
	average_loss = total_loss / steps
	accuracy = num_correct / data_size
	return average_loss, accuracy


def evaluation_loop(
	model: Module, loader: DataLoader, print_report: bool = True,
):
	model.eval()  # model needs to be in evaluation mode
	preds = []
	reals = []

	# turn off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode

		# initialize a list to store our predictions
		# loop over the test set
		for (x, y) in loader:
			# send the input to the device
			# x = x.to(device)

			# make the predictions and add them to the list
			pred = model(x)
			preds.extend(pred.argmax(axis=1).cpu().numpy())
			reals.extend(y.cpu().numpy())

	if print_report:
		# generate a classification report
		print(
			classification_report(
				y_true=np.array(reals),  # loader.dataset.targets.cpu().numpy(),
				y_pred=np.array(preds),
				target_names=loader.dataset.classes,
			)
		)
