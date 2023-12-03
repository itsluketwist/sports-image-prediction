import matplotlib

from utils import filename_datetime

from loop import History
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # set the backend so figures can be saved in the background


def plot_history(history: History, save_location: str = "output") -> None:
	plt.style.use("ggplot")
	fig = plt.figure(figsize=(12, 6))

	# plot loss on training and validation sets
	_ = fig.add_subplot(1, 2, 1)
	plt.title("Loss During Training")
	plt.plot(history.train_loss, label="training data")
	plt.plot(history.test_loss, label="validation data")
	plt.legend(facecolor="white", framealpha=1)
	plt.grid(True)
	plt.ylabel("loss")
	plt.xlabel("epoch")

	# plot accuracy on training and validation sets
	ax = fig.add_subplot(1, 2, 2)
	ax.yaxis.tick_right()
	ax.yaxis.set_label_position("right")
	plt.title("Accuracy During Training")
	plt.plot(history.train_accuracy, label="training data")
	plt.plot(history.test_accuracy, label="validation data")
	plt.legend(facecolor="white", framealpha=1)
	plt.grid(True)
	plt.ylabel("accuracy")
	plt.xlabel("epoch")

	plt.savefig(f"{save_location.rstrip('/')}/plot_{filename_datetime()}.png")
