from pathlib import Path
from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CONFIG = PROJECT_ROOT / "configs" / "sku110k.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"


def train():
    model = build_model()

    model.train(
        data=str(DATA_CONFIG),
        epochs=10,
        imgsz=640,
        batch=2,
        project=str(RUNS_DIR),
    )


if __name__ == "__main__":
    train()