from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "train-continued-3" / "weights" / "best.pt"
IMAGE_PATH = PROJECT_ROOT / "test_images" / "shelf.jpeg"

OUTPUT_DIR = PROJECT_ROOT / "predictions"


def predict():
    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="shelf_test",
        exist_ok=True,
    )

    print("Prediction finished.")

    for result in results:
        print("Detected objects:", len(result.boxes))


if __name__ == "__main__":
    predict()