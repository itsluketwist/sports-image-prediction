from typing import Literal

from torch import Tensor, flatten
from torch.nn import BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential


class LeNet5(Module):
    """
    An implementation of the LeNet-5 architecture.
    Further reading: https://en.wikipedia.org/wiki/LeNet
    """

    def __init__(self, num_classes: int, num_channels: Literal[1, 3] = 1):
        """
        Initialise the Lenet-5 model for image prediction.

        Must be used with (28x28) images as input.

        Parameters
        ----------
        num_classes: int
            How many possible classes does the input have?
        num_channels: Literal[1, 3]
            Will the input images be in colour?
        """
        # call the parent constructor
        super(LeNet5, self).__init__()

        # first group of convolutional layers
        self.layer_1 = Sequential(
            Conv2d(in_channels=num_channels, out_channels=6, kernel_size=5),
            BatchNorm2d(num_features=6),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )

        # second group of convolutional layers
        self.layer_2 = Sequential(
            Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            BatchNorm2d(num_features=16),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )

        # initialize fully connected layers
        self.fc_1 = Linear(in_features=256, out_features=120)
        self.relu_1 = ReLU()
        self.fc_2 = Linear(in_features=120, out_features=84)
        self.relu_2 = ReLU()

        # final output layer
        self.out = Linear(in_features=84, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        The feedforward pass of the Module.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
            The tensor after all layers have been applied.
        """
        # pass input through the convolutional layers
        x = self.layer_1(x)
        x = self.layer_2(x)

        # flatten the output after the convolutional layers
        x = flatten(x, 1)

        # pass through the fully-connected layers
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.relu_2(x)

        return self.out(x)
