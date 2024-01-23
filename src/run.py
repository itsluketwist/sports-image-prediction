import logging
import os
from argparse import ArgumentParser

from evaluate import run_evaluate
from predict import run_predict
from train import run_train
from utils import ModelOptions


SPORTS_PREDICTION_LOG_LEVEL = "SPORTS_PREDICTION_LOG_LEVEL"

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s : %(name)s - %(message)s",
    level=os.getenv(SPORTS_PREDICTION_LOG_LEVEL, logging.INFO),
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# create the main argument parser
parser = ArgumentParser()
parser.add_argument(
    "-m",
    "--model-type",
    type=ModelOptions,
    choices=list(ModelOptions),
    default=ModelOptions.SPORTS,
    help="Which model to use.",
)

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
    "-o",
    "--output-path",
    type=str,
    default="output",
    help="Path to output trained model weights and training figure.",
)
train_parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=10,
    help="How many epochs to train?",
)

# evaluate subparser
eval_parser = subparsers.add_parser(
    "evaluate",
    help="Evaluate a trained neural network.",
)
eval_parser.add_argument(
    "-i",
    "--input-path",
    type=str,
    required=True,
    help="What file to load the model from.",
)

# predict subparser
predict_parser = subparsers.add_parser(
    "predict",
    help="Use the pre-trained model to make predictions.",
)
predict_parser.add_argument(
    "-i",
    "--input-path",
    type=str,
    required=True,
    help="What file to load the model from.",
)
predict_parser.add_argument(
    "-n",
    "--number-samples",
    type=int,
    default=4,
    help="How many samples to predict.",
)
predict_parser.add_argument(
    "-s",
    "--sample-path",
    type=str,
    required=False,
    help="What file to load the sample image from.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    kwargs = vars(args)
    command = kwargs.pop("command")

    if command == "train":
        logger.info("Running TRAIN with kwargs: %s", kwargs)
        run_train(**kwargs)
    elif command == "evaluate":
        logger.info("Running EVALUATE with kwargs: %s", kwargs)
        run_evaluate(**kwargs)
    elif command == "predict":
        logger.info("Running PREDICT with kwargs: %s", kwargs)
        run_predict(**kwargs)
    else:
        logger.error("Incorrect command called: %s", command)
