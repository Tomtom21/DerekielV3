# Labeling Directory Structure

```
labeling/
├── frames/
│   ├── labeled/
│   │   ├── images/   # Labeled frame images
│   │   └── labels/   # Corresponding label files
│   └── unlabeled/
│       └── images/   # Unlabeled frame images
├── videos/
│   ├── processed/    # Videos that have been processed (frames parsed)
│   └── unprocessed/  # Videos waiting to be parsed and labeled
├── generated_createml/         # Generated CreateML training files
├── scripts/          # Scripts for frame splitting and CreateML generation
```

**Directory Descriptions:**

- **frames/**: Contains individual frames extracted from videos.
  - **labeled/**: Frames that have been labeled.
    - **images/**: Labeled frame images.
    - **labels/**: Corresponding label files.
  - **unlabeled/**: Frames that are not yet labeled.
    - **images/**: Unlabeled frame images.
- **videos/**: Raw video files.
  - **processed/**: Videos that have been processed (frames parsed).
  - **unprocessed/**: Videos waiting to be parsed and labeled.
- **generated_createml/**: Generated CreateML training files based on frames and labels.
- **scripts/**: Scripts for splitting frames from videos and generating CreateML files.
