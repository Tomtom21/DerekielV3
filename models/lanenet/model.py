from torch import nn, linspace
import torch.nn.functional as F
from torchvision import models

class LaneNet(nn.Module):
    """
    LaneNet Model using ResNet18 backbone:
    - ResNet18 feature extractor (pretrained, final layer removed)
    - Adaptive pooling (from ResNet)
    - Fully connected layers for 40 outputs (20 keypoints, each with x and visibility)
    """
    def __init__(self, num_rows=10, num_lanes=2, softmax_temp=0.2):
        super().__init__()

        self.num_rows = num_rows
        self.num_lanes = num_lanes
        self.num_kp = num_rows * num_lanes  # Total keypoints
        self.softmax_temp = softmax_temp
            
        # Use ResNet18 backbone, remove the final classification layer
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Getting the feature map extractions
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # Output: [batch, 512, 1, 1]
        
        # Getting one spatial map for each potential keypoint
        self.row_lane_head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_kp, kernel_size=1)
        )
        
        # Getting visibility information (not spatial)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.num_kp)
        )

    def forward(self, x):
        """
        Forward pass for LaneNet

        :param x: Input tensor
        """
        batch_size = x.size(0)

        # Feature extraction
        features = self.feature_extractor(x)  # [batch, 512, H, W]
    
        # Row-lane spatial maps, collapsing the height dimension
        maps = self.row_lane_head(features)  # [batch, num_kp, H, W]
        maps = maps.mean(dim=2)

        # Setting the fixed x positions to multiply against
        map_width = maps.size(2)
        fixed_x_coords = linspace(
            0.0, 1.0, map_width, device=maps.device
        ).view(1,1, map_width)  # [1, 1, W]

        # Running softmax for dist sum to 1, then multiply out to get position
        prob = F.softmax(maps / self.softmax_temp, dim=2)  # [batch, num_kp, W]
        x_positions = (prob * fixed_x_coords).sum(dim=2)  # [batch, num_kp]
        
        # Visibility head
        visibility_logits = self.visibility_head(features)  # [batch, num_kp]

        # Reshaping both to [batch, 2, 10] for 2 lanes, 10 rows each
        x_positions = x_positions.view(batch_size, self.num_lanes, self.num_rows)
        visibility_logits = visibility_logits.view(batch_size, self.num_lanes, self.num_rows)

        return x_positions, visibility_logits
