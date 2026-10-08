from ultralytics import YOLO

from paths import PROJECT_ROOT


def build_model():
    return YOLO(str(PROJECT_ROOT / "src/yolo11n.pt"))


if __name__ == "__main__":
    print(build_model())
