"""Start a new five-epoch training schedule from trained weights."""
import argparse
from pathlib import Path

from ultralytics import YOLO

from paths import DATA_CONFIG, PROJECT_ROOT, RUNS_DIR, require_file


def main():
    model_path = PROJECT_ROOT / "runs/train-2/weights/best.pt"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=model_path)
    args = parser.parse_args()
    model = YOLO(str(require_file(args.model)))

    model.train(
        data=str(require_file(DATA_CONFIG)),
        epochs=5,
        imgsz=640,
        batch=2,
        project=str(RUNS_DIR),
        name="train-continued",
        workers=0,
        resume=False,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
