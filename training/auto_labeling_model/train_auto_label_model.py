from ultralytics import YOLO
from pathlib import Path
from torch import backends

# Get the directory where the script is being called from
base_dir = Path(__file__).parent.resolve()

# Path to your dataset YAML file (YOLO format) inside 'dataset' directory
data_yaml_path = base_dir / 'dataset' / 'data.yaml'

# Output directory for training results inside the script directory
output_dir = base_dir / 'output'

# Model name (YOLOv8n)
model_name = 'yolov8n.pt'

def train_yolov8n():
    # Load YOLOv8n model
    model = YOLO(model_name)

    # Train the model
    model.train(
        data=data_yaml_path,
        epochs=100,           # Number of epochs, adjust as needed
        imgsz=400,     
        project=output_dir,   # Output directory
        name='yolov8n-custom', # Experiment name
        device='mps' if backends.mps.is_available() else 'cpu'
    )

if __name__ == '__main__':
    train_yolov8n()
