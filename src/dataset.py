from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "SKU-110K"

ANNOTATIONS_DIR = DATASET_DIR / "annotations"
LABELS_DIR = DATASET_DIR / "labels"


def xyxy_to_yolo(x1, y1, x2, y2, image_width, image_height):
    box_width = x2 - x1
    box_height = y2 - y1

    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2

    center_x /= image_width
    center_y /= image_height

    box_width /= image_width
    box_height /= image_height

    return center_x, center_y, box_width, box_height


def read_annotations(csv_path):
    annotations = {}

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:
            image_name = row[0]

            x1 = float(row[1])
            y1 = float(row[2])
            x2 = float(row[3])
            y2 = float(row[4])

            image_width = float(row[6])
            image_height = float(row[7])

            annotation = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "image_width": image_width,
                "image_height": image_height,
            }

            if image_name not in annotations:
                annotations[image_name] = []

            annotations[image_name].append(annotation)

    return annotations


def create_yolo_labels(annotations):
    LABELS_DIR.mkdir(exist_ok=True)

    for image_name, boxes in annotations.items():

        label_name = Path(image_name).with_suffix(".txt").name
        label_path = LABELS_DIR / label_name

        with open(label_path, "w", encoding="utf-8") as file:

            for box in boxes:

                x, y, width, height = xyxy_to_yolo(
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                    box["image_width"],
                    box["image_height"],
                )

                class_id = 0

                file.write(
                    f"{class_id} "
                    f"{x:.6f} "
                    f"{y:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )


def create_image_list(image_names, output_file):
    with open(output_file, "w", encoding="utf-8") as file:

        for image_name in image_names:
            file.write(f"./images/{image_name}\n")


def process_split(split_name):
    csv_path = ANNOTATIONS_DIR / f"annotations_{split_name}.csv"

    print(f"Processing {split_name}...")

    annotations = read_annotations(csv_path)

    create_yolo_labels(annotations)

    image_names = annotations.keys()

    output_file = DATASET_DIR / f"{split_name}.txt"

    create_image_list(image_names, output_file)

    print(
        f"{split_name}: {len(annotations)} images"
    )


def main():
    process_split("train")
    process_split("val")
    process_split("test")

    print()
    print("Dataset conversion finished.")


if __name__ == "__main__":
    main()