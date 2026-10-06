from pathlib import Path
import shutil
import tarfile
import urllib.request


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATASET_DIR = DATA_DIR / "SKU-110K"
ARCHIVE_PATH = DATA_DIR / "SKU110K_fixed.tar.gz"

DATASET_URL = (
    "http://trax-geometry.s3.amazonaws.com/"
    "cvpr_challenge/SKU110K_fixed.tar.gz"
)


def create_data_folder():
    DATA_DIR.mkdir(exist_ok=True)


def download_dataset():
    if ARCHIVE_PATH.exists():
        print("Archive already downloaded.")
        return

    print("Downloading SKU-110K...")
    urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)

    print("Download finished.")


def extract_dataset():
    if DATASET_DIR.exists():
        print("Dataset already extracted.")
        return

    print("Extracting dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        archive.extractall(DATA_DIR)

    extracted_folder = DATA_DIR / "SKU110K_fixed"

    if not extracted_folder.exists():
        raise FileNotFoundError(
            "Expected extracted folder SKU110K_fixed was not found."
        )

    extracted_folder.rename(DATASET_DIR)

    print("Extraction finished.")


def main():
    create_data_folder()
    download_dataset()
    extract_dataset()

    print()
    print("Dataset location:")
    print(DATASET_DIR)


if __name__ == "__main__":
    main()