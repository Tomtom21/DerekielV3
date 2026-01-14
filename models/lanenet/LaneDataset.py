import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import json
import pandas as pd
from pathlib import Path
import numpy as np
import random
from torchvision import transforms

class LaneDataset(Dataset):
    """
    Custom Dataset for Lane Detection
    """
    def __init__(
        self, 
        dataset_dir, 
        transform=None, 
        hflip_prob=0.2, 
        blur_prob=0.2
    ):
        # Getting the JSON and image dir from the dataset directory
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / 'images'
        self.annotation_file = self.dataset_dir / 'annotations.json'

        # Making sure the image directory exists
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Making sure the annotation file exists
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Load annotations
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.transform = transform
        self.hflip_prob = hflip_prob
        self.blur_prob = blur_prob

    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """
        return len(self.annotations)
    
    def _augment(self, image, placed_keypoints):
        """
        Apply random augmentations to the image and adjust keypoints accordingly.
        Returns the possibly augmented image and keypoints.
        """
        flipped = False
        # Horizontal flip
        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = True

        # Only blur (no noise)
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=1.25))

        # Adjust keypoints if flipped
        if flipped:
            flipped_keypoints = {}
            for key, x in placed_keypoints.items():
                if key.startswith('L_row_'):
                    new_key = key.replace('L_row_', 'R_row_')
                elif key.startswith('R_row_'):
                    new_key = key.replace('R_row_', 'L_row_')
                else:
                    new_key = key
                flipped_keypoints[new_key] = 1.0 - x
            placed_keypoints = flipped_keypoints

        # Applying color jitter
        image = transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.1, hue=0.1)(image)

        return image, placed_keypoints

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        :param idx: Index of the sample to retrieve
        """
        # Get annotation for the given index
        annotation = self.annotations[idx]
        image_name = annotation['file_upload']
        lane_labels = annotation['annotations'][0]['result']

        # Load image
        image_path = self.image_dir / image_name
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Making a look up table for available lane keypoints
        placed_keypoints = {}
        for item in lane_labels:
            label = item['value']
            if label and label.get('keypointlabels'):
                keypoint_label = label['keypointlabels'][0]
                if keypoint_label.startswith('L_row_') or keypoint_label.startswith('R_row_'):
                    x = label['x']/100.0
                    placed_keypoints[keypoint_label] = x

        # Data augmentation
        image, placed_keypoints = self._augment(image, placed_keypoints)

        # Apply user-provided transform (e.g., ToTensor, normalization)
        if self.transform:
            image = self.transform(image)

        # Generating our output tensor
        output = np.zeros(40, dtype=np.float32)  # 20 keypoints * 2 (x, visibility) = 40

        for side, offset in zip(['L_row', 'R_row'], [0, 10]):
            for keypoint_idx in range(10):
                keypoint_name = f"{side}_{keypoint_idx}"
                if keypoint_name in placed_keypoints:
                    x = placed_keypoints[keypoint_name]
                    visibility = 1.0
                else:
                    x = 0.0
                    visibility = 0.0
                output[(offset + keypoint_idx) * 2] = visibility
                output[(offset + keypoint_idx) * 2 + 1] = x

        # Convert output to tensor
        output_tensor = torch.tensor(output, dtype=torch.float32)

        return image, output_tensor
