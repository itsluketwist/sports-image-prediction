import logging

import matplotlib
import matplotlib.pyplot as plt
import torch

from constants import KMNIST_LABELS, SPORTS_LABELS
from load_data import MEAN, STD, get_kmnist_sample, get_sports_sample, load_sports_image
from utils import ModelOptions, get_device


logger = logging.getLogger(__name__)


def run_predict(
    input_path: str,
    sample_path: str | None = None,
    model_type: ModelOptions = ModelOptions.SPORTS,
    number_samples: int = 4,
):
    """
    Make predictions using the chosen model, image will be displayed with the result.

    Parameters
    ----------
    input_path: str
        Location of the model (.pth file) to use for prediction.
    sample_path: Optional[str] = None
        Location of a sports image to make predictions on.
    model: ModelOptions = ModelOptions.SPORTS
        Which model type is being evaluated.
    number_samples: int = 4
        If a path is not provided, how many samples from the dataset to make predictions on.
    """
    device = get_device()  # configure the device to use

    if model_type == ModelOptions.SPORTS:
        # core model
        classes = SPORTS_LABELS

        if sample_path:
            logger.debug("Loading sports image sample from path: %s", sample_path)
            _image, _label = load_sports_image(path=sample_path)
            _image = _image.to(_image)
            loader = [(_image, _label)]
        else:
            loader = get_sports_sample(count=number_samples)

    elif model_type == ModelOptions.KMNIST:
        # additional basic model
        classes = KMNIST_LABELS
        loader = get_kmnist_sample(count=number_samples)

        if sample_path:
            logger.warn(
                "Unsupported: KMNIST model cannot make predictions on a single image."
            )

    logger.debug("Loaded sample data for predictions.")

    # load the model and set it to evaluation mode
    loaded_model = torch.load(input_path).to(device)
    loaded_model.eval()
    logger.info("Loaded model and sample images - ready to predict!")

    matplotlib.use("TkAgg")

    # switch off autograd
    with torch.no_grad():
        # loop over samples
        for image, act_label in loader:
            # send the input to the device and make predictions on it
            image = image.to(device)
            pred_tensor = loaded_model(image)
            pred_label = classes[pred_tensor.argmax(axis=1).cpu().numpy()[0]]

            if not isinstance(act_label, str):
                act_label = classes[act_label]

            plt.figure(figsize=(5, 5))
            plt.xticks([])
            plt.yticks([])
            result = "✔️" if act_label == pred_label else "✖️"
            plt.title(f"Actual: {act_label} | Prediction: {pred_label} {result}")

            # prepare image
            image_tensor = image[0]
            if model_type == ModelOptions.SPORTS:
                # reverse earlier normalization for sports images
                image_tensor = image_tensor * STD[:, None, None] + MEAN[:, None, None]

            plt.imshow(image_tensor.permute(1, 2, 0))
            plt.show()
