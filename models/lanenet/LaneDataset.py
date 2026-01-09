import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd
from pathlib import Path
import numpy as np

class LaneDataset(Dataset):
    """
    Custom Dataset for Lane Detection
    """
    def __init__(self, dataset_dir, transform=None):
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

    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """
        return len(self.annotations)
    
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
        image = Image.open(image_path).convert("RGB")

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        # Making a look up table for available lane keypoints
        placed_keypoints = {}
        for item in lane_labels:
            label = item['value']
            if label and label.get('keypointlabels'):
                keypoint_label = label['keypointlabels'][0]
                if keypoint_label.startswith('L_row_') or keypoint_label.startswith('R_row_'):
                    x = label['x']/100.0
                    placed_keypoints[keypoint_label] = x
        
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
