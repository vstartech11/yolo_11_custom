from ultralytics import YOLO

model = YOLO('yolo11m.pt')
if __name__ == "__main__":
    model.train(data = "dataset_custom1.yaml", imgsz = 640, batch=16, epochs=150, device=0, cache=True, workers=4)



