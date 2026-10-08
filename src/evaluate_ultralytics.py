import argparse
from pathlib import Path

from ultralytics import YOLO

from paths import DATA_CONFIG, MODEL_PATH, RUNS_DIR, require_file


def main():
    parser = argparse.ArgumentParser(description="Validate on the SKU-110K val split.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DATA_CONFIG)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0:
        parser.error("Batch must be positive and workers nonnegative.")
    model = YOLO(str(require_file(args.model)))
    metrics = model.val(
        data=str(require_file(args.data)), split="val", imgsz=640,
        batch=args.batch, workers=args.workers, max_det=718,
        conf=0.001, iou=0.7, plots=True,
        project=str(RUNS_DIR / "detect"), name="val", exist_ok=False,
    )
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Results:", metrics.save_dir)


if __name__ == "__main__":
    main()
