from pathlib import Path

def create_structure(base_dir: Path):
    (base_dir / "frames" / "labeled" / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "frames" / "labeled" / "labels").mkdir(parents=True, exist_ok=True)
    (base_dir / "frames" / "unlabeled" / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "videos" / "processed").mkdir(parents=True, exist_ok=True)
    (base_dir / "videos" / "unprocessed").mkdir(parents=True, exist_ok=True)
    (base_dir / "generated_createml").mkdir(parents=True, exist_ok=True)
    (base_dir / "scripts").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Get the parent directory of the script's directory (i.e., the labeling directory)
    labeling_dir = Path(__file__).resolve().parent.parent
    create_structure(labeling_dir)

