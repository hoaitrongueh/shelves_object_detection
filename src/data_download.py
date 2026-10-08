from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

from paths import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "SKU-110K"
ARCHIVE_PATH = DATA_DIR / "SKU110K_fixed.tar.gz"
DATASET_URL = (
    "https://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
)


def create_data_folder():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():
    if ARCHIVE_PATH.exists():
        print("Archive already downloaded.")
        return
    partial_path = ARCHIVE_PATH.with_suffix(".gz.part")
    print("Downloading SKU-110K...")
    urllib.request.urlretrieve(DATASET_URL, partial_path)
    partial_path.rename(ARCHIVE_PATH)
    print("Download finished.")


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    members = archive.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        if (not target.is_relative_to(destination)
                or "\\" in member.name or ":" in member.name
                or not (member.isfile() or member.isdir())):
            raise ValueError(f"Unsafe archive member: {member.name}")
    archive.extractall(destination, members=members)


def extract_dataset():
    if DATASET_DIR.exists():
        print("Dataset already exists; leaving it unchanged.")
        return
    with tempfile.TemporaryDirectory(dir=DATA_DIR) as temporary_dir:
        with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
            safe_extract(archive, temporary_dir)
        extracted = Path(temporary_dir) / "SKU110K_fixed"
        if not extracted.is_dir():
            raise FileNotFoundError("Expected archive folder SKU110K_fixed.")
        # The original archive has flat images named train_*, val_*, test_*.
        images_dir = extracted / "images"
        for image_path in list(images_dir.iterdir()):
            if not image_path.is_file():
                continue
            split_name = image_path.name.split("_", 1)[0]
            if split_name not in ("train", "val", "test"):
                raise ValueError(f"Cannot determine image split: {image_path.name}")
            split_dir = images_dir / split_name
            split_dir.mkdir(exist_ok=True)
            image_path.rename(split_dir / image_path.name)
        if DATASET_DIR.exists():
            raise FileExistsError(f"Dataset appeared during extraction: {DATASET_DIR}")
        shutil.move(str(extracted), str(DATASET_DIR))
    print("Extraction finished:", DATASET_DIR)


def main():
    create_data_folder()
    if DATASET_DIR.exists():
        print("Dataset already exists; leaving it unchanged:", DATASET_DIR)
        return
    download_dataset()
    extract_dataset()


if __name__ == "__main__":
    main()
