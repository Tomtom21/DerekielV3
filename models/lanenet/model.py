from torch import nn
from torchvision import models

class LaneNet(nn.Module):
    """
    LaneNet Model using ResNet18 backbone:
    - ResNet18 feature extractor (pretrained, final layer removed)
    - Adaptive pooling (from ResNet)
    - Fully connected layers for 40 outputs (20 keypoints, each with x and visibility)
    """
    def __init__(self):
        super().__init__()
        # Use ResNet18 backbone, remove the final classification layer
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # Output: [batch, 512, 1, 1]
        self.conv_head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 40, kernel_size=1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass for LaneNet

        :param x: Input tensor
        """
        x = self.feature_extractor(x)
        x = self.conv_head(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 40]
        return x
