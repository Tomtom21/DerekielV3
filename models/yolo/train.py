from pathlib import Path

from ultralytics import YOLO
from torch import backends

# Getting the dataset directory for YOLO
dataset_dir = Path(__file__).parent.parent.parent / 'data' / 'yolo_dataset'

# Defining paths
data_yaml_path = dataset_dir / 'data.yaml'
images_dir = dataset_dir / 'images'
labels_dir = dataset_dir / 'labels'
classes_file = dataset_dir / 'classes.txt'

# Model name
model_name = 'yolov8n.pt'

def train_yolo():
    # Load YOLO model
    model = YOLO(model_name)

    # Training the model
    model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=400,
        project=Path(__file__).parent / 'output',
        name=f'{model_name.split(".")[0]}-custom',
        device='mps' if backends.mps.is_available() else 'cpu'
    )

if __name__ == "__main__":
    train_yolo()
