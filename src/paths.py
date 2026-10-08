from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "runs/train-continued-3/weights/best.pt"
DATA_CONFIG = PROJECT_ROOT / "configs/sku110k.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"


def require_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path
