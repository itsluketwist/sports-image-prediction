import logging

import torch

from constants import KMNIST_LABELS, SPORTS_LABELS
from load_data import get_kmnist_eval_data, get_sports_eval_data
from loop import evaluation_loop
from utils import ModelOptions, get_device


logger = logging.getLogger(__name__)


def run_evaluate(
    input_path: str,
    model_type: ModelOptions = ModelOptions.SPORTS,
):
    """
    Evaluate a PyTorch model, printing the classification report.

    Parameters
    ----------
    input_path: str
        Location of the model (.pth file) to be evaluated.
    model: ModelOptions = ModelOptions.SPORTS
        Which model type is being evaluated.
    """
    device = get_device()  # configure the device to use
    logger.info("Loading model...")
    loaded_model = torch.load(input_path).to(device)
    logger.info("Model loaded!")

    # handle model-specific choices
    if model_type == ModelOptions.SPORTS:
        # core model
        logger.info("Getting sports data...")
        classes = SPORTS_LABELS
        eval_loader = get_sports_eval_data()

    elif model_type == ModelOptions.KMNIST:
        # additional basic model - used for learning and initial configuration
        logger.info("Getting kmnist data...")
        classes = KMNIST_LABELS
        eval_loader = get_kmnist_eval_data()

    else:
        logger.error("Invalid model type provided: %s", model_type)
        return
    logger.info("Data loaded!")

    logger.info("Evaluating network...")
    evaluation_loop(
        model=loaded_model,
        loader=eval_loader,
        classes=classes,
    )
