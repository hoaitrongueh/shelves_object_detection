from ultralytics import YOLO
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]

    model_path = project_root / "runs" / "train-2" / "weights" / "best.pt"
    data_path = project_root / "configs" / "sku110k.yaml"

    model = YOLO(model_path)

    model.train(
        data=data_path,
        epochs=5,
        imgsz=640,
        batch=2,
        project=project_root / "runs",
        name="train-continued",
        workers=8,
    )


if __name__ == "__main__":
    main()