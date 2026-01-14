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
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Output: [batch, 512, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 40)
        )

    def forward(self, x):
        """
        Forward pass for LaneNet

        :param x: Input tensor
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 512]
        x = self.fc(x)
        return x
