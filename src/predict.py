import argparse
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

from paths import MODEL_PATH, PROJECT_ROOT, require_file
from visualization import draw_boxes


def predict(image_path, model_path=MODEL_PATH, confidence=0.25, iou=0.5,
            thickness=4, output_dir=PROJECT_ROOT / "predictions"):
    model = YOLO(str(require_file(model_path)))
    with Image.open(require_file(image_path)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    result = model.predict(
        source=image, conf=confidence, iou=iou, imgsz=640,
        max_det=718, verbose=False,
    )[0]
    boxes = result.boxes.xyxy.cpu().tolist()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_path).stem}_prediction.png"
    number = 2
    while output_path.exists():
        output_path = output_dir / f"{Path(image_path).stem}_prediction_{number}.png"
        number += 1
    draw_boxes(image, boxes, thickness).save(output_path)
    print("Detected objects:", len(boxes))
    print("Saved prediction:", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Detect generic products in an image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--thickness", type=int, default=4)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "predictions")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        parser.error("Confidence and IoU must be between 0 and 1.")
    if args.thickness < 1:
        parser.error("Thickness must be at least 1.")
    predict(args.image, args.model, args.confidence, args.iou,
            args.thickness, args.output)


if __name__ == "__main__":
    main()
