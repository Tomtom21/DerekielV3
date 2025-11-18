"""
 Generates a CreateML dataset from the labeled frames
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
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to CreateML format.")
    parser.add_argument(
        "--yolo-dir",
        type=str,
        required=True,
        help="Name or path of the desired YOLO directory in labelstudio_exports."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parsing the command line arguments
    args = parse_args()
    yolo_dir = args.yolo_dir

    # Setting the dirs we want
    LABELING_DIR = Path(__file__).parent.parent
    YOLO_DIR = LABELING_DIR / "labelstudio_exports" / "yolo" / args.yolo_dir
    CREATEML_DIR = LABELING_DIR / "generated_createml" / "yolo"

    # Our list of converted annotations for CreateML
    annotations = []

    # Getting our list of image files/labels
    image_filenames = sorted(os.listdir(YOLO_DIR / "images"))
    label_filenames = sorted(os.listdir(YOLO_DIR / "labels"))

    # Getting the list of classes from the directory
    with open(YOLO_DIR / "classes.txt", "r") as classes_file:
        classes_list = [line.strip() for line in classes_file.read().splitlines()]

    classes = {i: name for i, name in enumerate(classes_list)}

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

        # Converting each text label into a label object we can work with
        yolo_labels = []    
        for line in label_lines:
            label = YOLO_Label(line, classes)
            yolo_labels.append(label)

        # Creating the CreateML annotation for this image
        image_annotation = {
            "image": image_filename,
            "annotations": [
                {
                    "label": label.name,
                    "coordinates": {
                        "x": label.x * image_width,
                        "y": label.y * image_height,
                        "width": label.width * image_width,
                        "height": label.height * image_height
                    }
                }
                for label in yolo_labels
            ]
        }
        annotations.append(image_annotation)

    # Writing out the CreateML annotations to a timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CREATEML_RUN_DIR = CREATEML_DIR / f"createml_{timestamp}"
    CREATEML_RUN_DIR.mkdir(parents=True, exist_ok=True)

    createml_annotations_path = CREATEML_RUN_DIR / "annotations.json"
    with open(createml_annotations_path, "w") as outfile:
        json.dump(annotations, outfile, indent=4)

    # Copy all images into the same directory as the annotation file
    images_src_dir = YOLO_DIR / "images"
    images_dst_dir = CREATEML_RUN_DIR
    for image_filename in image_filenames:
        src = images_src_dir / image_filename
        dst = images_dst_dir / image_filename
        shutil.copy2(src, dst)
