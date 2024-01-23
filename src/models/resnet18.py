from typing import Literal

from torch import Tensor, flatten
from torch.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torchvision.models import ResNet18_Weights, resnet18


def get_resnet_18_model(num_classes: int, use_pretrained: bool = True) -> Module:
    if use_pretrained is True:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = Linear(512, num_classes)
    else:
        model = ResNet18(num_classes=num_classes)

    return model


# notes: https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
# also: https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/


class ResidualBlock(Module):
    """A group of convolutional layers with a residual connection, use to build ResNet models."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Module | None = None,
        expansion: int = 1,
    ):
        """
        Initialise the residual block for use in a Module.

        Parameters
        ----------
        in_channels: int
            Number of channels input into the block.
        out_channels: int
            Number of channels output from the block.
        stride: int = 1
            The stride length to use.
        downsample: Module = None
            Whether to apply a down-sampling Module during the residual connection.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_channels),
            ReLU(),
        )
        self.conv2 = Sequential(
            Conv2d(
                out_channels,
                out_channels * expansion,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = ReLU()
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        The feed-forward pass of the Module.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
            The tensor after all layers have been applied.
        """
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(Module):
    """
    An implementation of the LeNet-5 architecture.
    Further reading: https://en.wikipedia.org/wiki/Residual_neural_network
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: Literal[1, 3] = 3,
        block: Module = ResidualBlock,
    ):
        """
        Initialise the ResNet-18 model for image prediction.

        Must be used with (224x224) images as input.

        Parameters
        ----------
        num_classes: int
            How many possible classes does the input have?

        Parameters
        ----------
        num_classes: int
            How many possible classes does the input have?
        num_channels: Literal[1, 3] = 3
            Will the input images be in colour?
        block: Module = ResidualBlock
            The type of layer-block to use when creating the Model.
        """
        # call the parent constructor
        super(ResNet18, self).__init__()

        # initial convolutional layers
        self.in_channels = 64
        self.pre_conv = Sequential(
            Conv2d(
                in_channels=num_channels,
                out_channels=self.in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            BatchNorm2d(self.in_channels),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # create groups of residual blocks (of layers)
        self.layer_1 = self._make_block_group(
            block=block,
            out_channels=64,
            num_blocks=2,
            first_stride=1,
        )
        self.layer_2 = self._make_block_group(
            block=block,
            out_channels=128,
            first_stride=2,
        )
        self.layer_3 = self._make_block_group(
            block=block,
            out_channels=256,
            first_stride=2,
        )
        self.layer_4 = self._make_block_group(
            block=block,
            out_channels=512,
            first_stride=2,
        )

        self.avg_pool = AvgPool2d(
            kernel_size=7,
            stride=1,
        )

        # final output layer
        self.out = Linear(in_features=512, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        The feed-forward pass of the Module.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
            The tensor after all layers have been applied.
        """
        x = self.pre_conv(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.avg_pool(x)
        x = flatten(x, 1)

        return self.out(x)

    def _make_block_group(
        self,
        block: Module,
        out_channels: int,
        first_stride: int,
        num_blocks: int = 2,
    ):
        """
        Make a group a residual blocks of layers.

        Parameters
        ----------
        block: Module
            _description_
        out_channels: int
            Number of channels output from the block.
        first_stride: int
            The stride length to use for the first block (others will have stride length 1).
        num_blocks: int = 2
            How many blocks to include in the group.

        Returns
        -------
        _type_
            _description_
        """
        if first_stride != 1 or self.in_channels != out_channels:
            # define the down-sampling layers if required for layer group
            downsample = Sequential(
                Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=first_stride,
                    bias=False,
                ),
                BatchNorm2d(out_channels),
            )
        else:
            downsample = None

        layers = []

        # first layer, may have different stride and include down-smapling layers
        layers.append(
            block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=first_stride,
                downsample=downsample,
            )
        )

        # add remaining layers
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                )
            )

        self.in_channels = out_channels  # configure in channels for next blocks

        return Sequential(*layers)
