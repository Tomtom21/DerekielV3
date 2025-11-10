# Labeling Directory Structure

```
labeling/
├── generated_createml/          # Generated CreateML training files
├── labelstudio_exports/         # Exports from Label Studio annotation tool
├── readme.md                    # Project documentation (this file)
├── scripts/                     # Scripts for data processing and conversion
│   ├── createml_converter.py            # Converts data to CreateML format
│   ├── generate_cropped_speed_signs.py  # Generates cropped images of speed signs
│   ├── generate_cropped_vehicle_images.py # Generates cropped images of vehicles
│   ├── setup_structure.py               # Script to set up this directory structure
│   └── split_frames_from_video.py       # Splits frames from videos
├── unlabeled_frames/             # Frames not yet labeled, organized by task
│   ├── speed_sign_classification      # Frames for speed sign classification
│   ├── vehicle_type_classification    # Frames for vehicle type classification
│   └── yolo                          # Frames for YOLO-based tasks
└── videos/                      # Raw video files
    ├── processed/                   # Videos that have been processed (frames parsed)
    └── unprocessed/                 # Videos waiting to be parsed and labeled
```

**Directory Descriptions:**

- **generated_createml/**: Contains training files generated for Apple's CreateML.
- **labelstudio_exports/**: Stores exports from the Label Studio annotation tool.
- **readme.md**: Documentation for the labeling directory structure and usage.
- **scripts/**: Python scripts for processing videos, generating cropped images, converting formats, and setting up the directory structure.
  - **createml_converter.py**: Converts labeled data to CreateML format.
  - **generate_cropped_speed_signs.py**: Extracts and saves cropped images of speed signs from frames.
  - **generate_cropped_vehicle_images.py**: Extracts and saves cropped images of vehicles from frames.
  - **setup_structure.py**: Script to create the directory structure as described here.
  - **split_frames_from_video.py**: Splits video files into individual frames.
- **unlabeled_frames/**: Contains frames that have not yet been labeled, organized by classification task or labeling method.
  - **speed_sign_classification/**: Frames for speed sign classification tasks.
  - **vehicle_type_classification/**: Frames for vehicle type classification tasks.
  - **yolo/**: Frames intended for YOLO-based object detection tasks.
- **videos/**: Raw video files used as sources for frame extraction.
  - **processed/**: Videos that have already been processed (frames extracted).
  - **unprocessed/**: Videos awaiting processing and labeling.
