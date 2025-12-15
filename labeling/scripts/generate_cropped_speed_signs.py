"""
Generating cropped images of speed signs for classification training.
"""
import os
import json
from pathlib import Path
import argparse
from PIL import Image
from datetime import datetime
import shutil

from YOLO_Label import YOLO_Label

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Generate cropped images of speed signs.")
    parser.add_argument(
        "--yolo-dir",
        type=str,
        required=True,
        help="Name or path of the desired YOLO directory in labelstudio_exports/yolo."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parsing the command line arguments
    args = parse_args()
    yolo_dir = args.yolo_dir

    # Setting the dirs we want
    LABELING_DIR = Path(__file__).parent.parent
    YOLO_DIR = LABELING_DIR / "labelstudio_exports" / "yolo" / args.yolo_dir
    CROPPED_SPEEDSIGNS_DIR = LABELING_DIR / "unlabeled_frames" / "speed_sign_classification"
    CROPPED_SPEEDSIGNS_DIR.mkdir(parents=True, exist_ok=True)

    # Getting our list of image files/labels
    image_filenames = sorted(os.listdir(YOLO_DIR / "images"))
    label_filenames = sorted(os.listdir(YOLO_DIR / "labels"))

    # Getting the list of classes from the directory
    with open(YOLO_DIR / "classes.txt", "r") as classes_file:
        classes_list = [line.strip() for line in classes_file.read().splitlines()]

    classes = {i: name for i, name in enumerate(classes_list)}

    # Looping through the images/labels
    for image_filename, label_filename in zip(image_filenames, label_filenames):
        # Loading the image
        image_path = YOLO_DIR / "images" / image_filename
        image = Image.open(image_path)
        image_width, image_height = image.size

        # Loading the label file
        label_path = YOLO_DIR / "labels" / label_filename
        with open(label_path, "r") as label_file:
            label_content = label_file.read()
            label_lines = label_content.splitlines()
        
        # Converting each text label into a label object
        yolo_labels = []
        for line in label_lines:
            label = YOLO_Label(line, classes)
            yolo_labels.append(label)

        # Filtering the labels down to just speed signs
        speed_sign_labels = [label for label in yolo_labels if label.name == "speedsign"]

        # Cropping and saving each of the speed sign images
        for idx, label in enumerate(speed_sign_labels):
            # Calculating the bounding box in pixel coordinates
            x_center = label.x * image_width
            y_center = label.y * image_height
            box_width = label.width * image_width
            box_height = label.height * image_height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Cropping the image
            cropped_image = image.crop((x1, y1, x2, y2))

            # Saving the cropped image
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            cropped_image_filename = f"{image_filename.split('.')[0]}_speed_sign_{idx}.png"
            cropped_image_path = CROPPED_SPEEDSIGNS_DIR / cropped_image_filename
            cropped_image.save(cropped_image_path)
