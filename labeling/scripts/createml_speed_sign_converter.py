"""
Converts a YOLO formatted dataset of classification data into CreateML Format
"""
import os
import json
from datetime import datetime
from pathlib import Path
import argparse
import shutil

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to CreateML format.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Name or path of the desired YOLO directory in labelstudio_exports."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parsing the command line arguments
    args = parse_args()
    input_dir = args.input_dir

    # Setting the dirs we want
    LABELING_DIR = Path(__file__).parent.parent
    INPUT_DIR = LABELING_DIR / "labelstudio_exports" / "speed_sign_classification" / args.input_dir
    CREATEML_DIR = LABELING_DIR / "generated_createml" / "speed_sign_classification"
    CREATEML_DIR.mkdir(parents=True, exist_ok=True)

    # Getting our list of image files
    image_filenames = sorted(os.listdir(INPUT_DIR / "images"))

    # Getting the data from the json file
    with open(INPUT_DIR / "labels.json", "r") as json_file:
        labels_data = json.load(json_file)

    # Making sure the destination directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CREATEML_SAVE_DIR = CREATEML_DIR / f"createml_{timestamp}"
    CREATEML_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Looping through our labels and copying the images into the CreateML format
    for label in labels_data:
        image_filename = label["file_upload"]
        class_name = label["annotations"][0]["result"][0]["value"]["choices"][0]

        # Source and destination paths
        src_image_path = INPUT_DIR / "images" / image_filename
        dest_class_dir = CREATEML_SAVE_DIR / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        # Copying the image to the new location
        dest_image_path = dest_class_dir / image_filename
        shutil.copy(src_image_path, dest_image_path)
