from ultralytics import YOLO

def build_model():
    model=YOLO('yolo11n.pt')
    return model
if __name__=="__main__":
    model=build_model()
    print(model)