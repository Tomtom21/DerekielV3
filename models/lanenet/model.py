from torch import nn
from torchvision import models

class LaneNet(nn.Module):
    """
    LaneNet Model using ResNet18 backbone:
    - ResNet18 feature extractor (pretrained, final layer removed)
    - Adaptive pooling (from ResNet)
    - Fully connected layers for 40 outputs (20 keypoints, each with x and visibility)
    """
    def __init__(self, num_rows=10, points_per_row=2):
        super().__init__()
        # Use ResNet18 backbone, remove the final classification layer
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # Output: [batch, 512, 1, 1]
        
        self.conv_reduction = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        # Trunk fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.x_position_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 20),  # 20 x-positions
            nn.Sigmoid()
        )

        self.visibility_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 20)  # 20 visibility scores
            #activation function??
        )


    def forward(self, x):
        """
        Forward pass for LaneNet

        :param x: Input tensor
        """
        # Getting the batch size
        batch_size = x.size(0)

        # Feature extraction
        features = self.feature_extractor(x)  # [batch, 512, H, W]

        # Convolutions
        conv_features = self.conv_reduction(features)  # [batch, 20, H', W']

        # Flattening for fully connected layers
        flattened_features = conv_features.flatten(start_dim=1) # [batch, 128, 4, 8] -> [batch, 4096]

        # FC Trunk
        trunk = self.shared_fc(flattened_features)  # [batch, 512]

        # X position and visibility heads
        x_positions = self.x_position_head(trunk)  # [batch, 20]
        visibility_scores = self.visibility_head(trunk)  # [batch, 20]

        # Reshaping to [batch, 2, 10]
        x_positions = x_positions.view(batch_size, 2, 10)
        visibility_scores = visibility_scores.view(batch_size, 2, 10)

        return x_positions, visibility_scores
