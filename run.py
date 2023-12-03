from argparse import ArgumentParser
import logging
import os

from src.train import run_train
from src.predict import run_predict


SPORTS_PREDICTION_LOG_LEVEL = "SPORTS_PREDICTION_LOG_LEVEL"

logging.basicConfig(
	format="[%(asctime)s] %(levelname)-8s : %(name)s - %(message)s",
	level=os.getenv(SPORTS_PREDICTION_LOG_LEVEL, logging.INFO),
	datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# create the main argument parser
parser = ArgumentParser()
subparsers = parser.add_subparsers(
	dest="command",
	help="sub-command help",
)

# train subparser
train_parser = subparsers.add_parser(
	"train",
	help="Train the neural network from scratch.",
)
train_parser.add_argument(
	"-m",
	"--model-path",
	type=str,
	default="output",
	help="Path to output trained model weights.",
)
train_parser.add_argument(
	"-p",
	"--plot-path",
	type=str,
	default="output",
	help="Path to output training data figure.",
)

# predict subparser
predict_parser = subparsers.add_parser(
	"predict",
	help="Use the pre-trained model to make predictions.",
)
predict_parser.add_argument(
	"-m",
	"--model-path",
	type=str,
	required=True,
	help="Where to load the model from.",
)
predict_parser.add_argument(
	"-n",
	"--number-samples",
	type=int,
	default=10,
	help="How many samples to predict.",
)


if __name__ == "__main__":
	args = parser.parse_args()
	kwargs = vars(args)
	command = kwargs.pop("command")

	if command == "train":
		logger.info("Running TRAIN with kwargs: %s", kwargs)
		run_train(**kwargs)
	elif command == "predict":
		logger.info("Running PREDICT with kwargs: %s", kwargs)
		run_predict(**kwargs)
	else:
		logger.error("Incorrect command called: %s", command)
