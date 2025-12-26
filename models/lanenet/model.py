from torch import nn

class ConvBlock(nn.Module):
    """
    Convolution Block for LaneNet Model
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass for ConvBlock

        :param x: Input tensor
        """
        return self.block(x)

class LaneNet(nn.Module):
    """
    LaneNet Model that follows the following structure:
    - Convolutional layers using ConvBlocks
    - Adaptive Average Pooling
    - 40 Fully connected layers
        - 20 keypoints for each lane, with each pair of 2 being x and visibility
    """
    def __init__(self):
        super().__init__()

        # Defining convolution layers using ConvBlocks
        self.conv = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2)
        )

        # Flattening laye.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 40)
        )

    def forward(self, x):
        """
        Forward pass for LaneNet

        :param x: Input tensor
        """
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
