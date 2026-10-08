import argparse
import csv
from pathlib import Path

from paths import PROJECT_ROOT


DATASET_DIR = PROJECT_ROOT / "data/SKU-110K"


def xyxy_to_yolo(x1, y1, x2, y2, image_width, image_height):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError("Box corners must be ordered and inside the image.")
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + box_width / 2) / image_width
    center_y = (y1 + box_height / 2) / image_height
    return center_x, center_y, box_width / image_width, box_height / image_height


def read_annotations(csv_path):
    annotations = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        for line_number, row in enumerate(csv.reader(file), 1):
            if not row:
                continue
            try:
                image_name = row[0]
                if Path(image_name).name != image_name or "\\" in image_name:
                    raise ValueError("Expected an image filename, not a path.")
                box = dict(
                    x1=float(row[1]), y1=float(row[2]),
                    x2=float(row[3]), y2=float(row[4]),
                    image_width=float(row[6]), image_height=float(row[7]),
                )
                xyxy_to_yolo(**box)
            except (IndexError, ValueError) as error:
                raise ValueError(f"{csv_path}:{line_number}: {error}") from error
            annotations.setdefault(image_name, []).append(box)
    return annotations


def create_yolo_labels(annotations, labels_dir, overwrite=False):
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_name, boxes in annotations.items():
        label_path = labels_dir / Path(image_name).with_suffix(".txt").name
        lines = []
        for box in boxes:
            x, y, width, height = xyxy_to_yolo(**box)
            lines.append(f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n")
        content = "".join(lines)
        if label_path.exists() and not overwrite:
            if label_path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"Existing label differs: {label_path}; "
                    "use --overwrite deliberately."
                )
            continue
        label_path.write_text(content, encoding="utf-8")


def create_image_list(image_names, output_file, split_name):
    with Path(output_file).open("w", encoding="utf-8") as file:
        for image_name in image_names:
            file.write(f"./images/{split_name}/{image_name}\n")


def process_split(split_name, dataset_dir=DATASET_DIR, overwrite=False):
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "annotations" / f"annotations_{split_name}.csv"
    annotations = read_annotations(csv_path)
    image_dir = dataset_dir / "images" / split_name
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Arrange images in {image_dir} before conversion.")
    for image_name in annotations:
        if not (image_dir / image_name).is_file():
            raise FileNotFoundError(image_dir / image_name)
    create_yolo_labels(annotations, dataset_dir / "labels" / split_name, overwrite)
    create_image_list(annotations, dataset_dir / f"{split_name}.txt", split_name)
    print(f"{split_name}: {len(annotations)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SKU-110K CSV boxes to YOLO labels."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace differing label files."
    )
    args = parser.parse_args()
    for split_name in ("train", "val", "test"):
        process_split(split_name, args.dataset, args.overwrite)
    print("Dataset conversion finished.")


if __name__ == "__main__":
    main()
