from model import build_model
from paths import DATA_CONFIG, RUNS_DIR, require_file


def train():
    model = build_model()

    model.train(
        data=str(require_file(DATA_CONFIG)),
        epochs=10,
        imgsz=640,
        batch=2,
        project=str(RUNS_DIR),
        workers=0,
        exist_ok=False,
    )


if __name__ == "__main__":
    train()
