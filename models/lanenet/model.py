from torch import nn, linspace
from torchvision import models
import torch.nn.functional as F

class LaneNet(nn.Module):
    """
    LaneNet Model using ResNet18 backbone:
    - ResNet18 feature extractor (pretrained, final layer removed)
    - Adaptive pooling (from ResNet)
    - Fully connected layers for 40 outputs (20 keypoints, each with x and visibility)
    """
    def __init__(self, num_rows=10, points_per_row=2):
        super().__init__()
        self.num_rows = num_rows
        self.points_per_row = points_per_row

        # Use ResNet18 backbone, remove the final classification layer
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # Output: [batch, 512, 1, 1]
        self.backbone_processing = nn.Sequential(
            backbone.conv1,   # stride 2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool, # stride 2 (total 4)
            backbone.layer1,  # stride 4
            backbone.layer2,  # stride 8
        )

        self.feature_refine = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, points_per_row, kernel_size=1),
        )

    def forward(self, x):
        """
        Forward pass for LaneNet

        :param x: Input tensor
        """
        # Getting the batch size
        batch_size = x.size(0)

        # Feature extraction
        features = self.backbone_processing(x)  # [batch, 512, H, W]

        # Convolutions/pooling
        heatmaps = self.feature_refine(features)  # [batch, 20, H', W']
        heatmaps = F.interpolate(
            heatmaps,
            size=(self.num_rows, heatmaps.shape[-1]),
            mode="bilinear",
            align_corners=False
        )

        # Decode to match dataset format
        x_positions, visibility_scores = self._decode_heatmaps(heatmaps)
        x_positions = x_positions.view(batch_size, 2, 10)

        visibility_scores = visibility_scores.view(batch_size, 2, 10)

        return x_positions, visibility_scores

    def _decode_heatmaps(self, heatmaps):
        """
        heatmaps: [B, 2, 10, W]
        """
        # Softmax over width
        probs = F.softmax(heatmaps, dim=-1)

        W = heatmaps.shape[-1]
        idx = linspace(0, 1, W, device=heatmaps.device)

        # Soft-argmax
        x_positions = (probs * idx).sum(dim=-1)

        # Visibility = peak confidence per row
        visibility_scores = heatmaps.max(dim=-1).values

        return x_positions, visibility_scores