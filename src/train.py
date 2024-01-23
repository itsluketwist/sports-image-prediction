import logging

import torch
from torch import nn
from torch.optim import Adam

from constants import KMNIST_LABELS, SPORTS_LABELS
from load_data import (
    get_kmnist_eval_data,
    get_kmnist_train_data,
    get_sports_eval_data,
    get_sports_train_data,
)
from loop import evaluation_loop, testing_loop, training_loop
from models.lenet5 import LeNet5
from models.resnet18 import get_resnet_18_model
from plot import plot_history
from utils import History, ModelOptions, filename_datetime, get_device


logger = logging.getLogger(__name__)


# default training hyperparameters
DEFAULT_INIT_LEARN_RATE = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_TRAIN_SPLIT = 0.75


def run_train(
    model_type: ModelOptions = ModelOptions.SPORTS,
    init_learning_rate: float = DEFAULT_INIT_LEARN_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    output_path: str = "output",
    use_pretrained: bool = True,
):
    """
    Train the chosen model, given the hyperparameters.

    Parameters
    ----------
    model_type: ModelOptions = ModelOptions.SPORTS
        Which model type is being trained.
    init_learning_rate: float = DEFAULT_INIT_LEARN_RATE
        Learning rate to use for training.
    batch_size: int = DEFAULT_BATCH_SIZE
        Batch size to use during training.
    epochs: int = DEFAULT_EPOCHS
        How many epochs to train for.
    train_split: float = DEFAULT_TRAIN_SPLIT
        What percentage of data to use for training vs. testing.
    output_path: str = "output"
        Location to save the model (.pth file) and learning curves (.png file) after training.
    use_pretrained: bool = True
        Whether to use a pretrained ResNet18 model, or train one from scratch.
    """
    device = get_device()  # configure the device to use

    # handle model-specific choices
    if model_type == ModelOptions.SPORTS:
        # core model
        logger.info("Getting sports data...")
        classes = SPORTS_LABELS
        train_loader, test_loader = get_sports_train_data(
            batch_size=batch_size,
            use_percent=1,
        )
        eval_loader = get_sports_eval_data()

        logger.info("Initializing the ResNet model...")
        cnn_model = get_resnet_18_model(
            num_classes=len(classes),
            use_pretrained=use_pretrained,
        )

    elif model_type == ModelOptions.KMNIST:
        # additional basic model - used for learning and initial configuration
        logger.info("Getting kmnist data...")
        classes = KMNIST_LABELS
        train_loader, test_loader = get_kmnist_train_data(
            train_split=train_split,
            batch_size=batch_size,
        )
        eval_loader = get_kmnist_eval_data()

        logger.info("Initializing the LeNet model...")
        cnn_model = LeNet5(
            num_classes=len(classes),
        )

    else:
        logger.error("Invalid model type provided: %s", model_type)
        return

    loss_func = nn.CrossEntropyLoss()  # init loss function
    opt = Adam(cnn_model.parameters(), lr=init_learning_rate)  # init optimizer
    hist = History()  # init history dict
    cnn_model = cnn_model.to(device)

    logger.info("Beginning to train the network...")
    for e in range(0, epochs):
        train_loss, train_accuracy = training_loop(
            model=cnn_model,
            loader=train_loader,
            loss_func=loss_func,
            optimizer=opt,
        )
        test_loss, test_accuracy = testing_loop(
            model=cnn_model,
            loader=test_loader,
            loss_func=loss_func,
        )
        hist.update(
            train_loss=train_loss.cpu().detach().numpy(),
            train_accuracy=train_accuracy,
            test_loss=test_loss.cpu().detach().numpy(),
            test_accuracy=test_accuracy,
        )

        # print this loops training and testing data
        logger.info("EPOCH: %s/%s", e + 1, epochs)
        logger.info(
            f"Train loss: {train_loss:.6f}, Train accuracy: {train_accuracy:.4f}",
        )
        logger.info(
            f"Val loss: {test_loss:.6f}, Val accuracy: {test_accuracy:.4f}\n",
        )

    # training complete, save the model to disk
    torch.save(
        obj=cnn_model,
        f=f"{output_path.rstrip('/')}/{model_type}_mod_{filename_datetime()}.pth",
    )

    # create a figure from the training data
    plot_history(
        history=hist,
        model_name=model_type.value,
        save_location=output_path,
    )

    # evaluate the model on the evaluation dataset
    logger.info("Evaluating network...")
    evaluation_loop(
        model=cnn_model,
        loader=eval_loader,
        classes=classes,
    )
