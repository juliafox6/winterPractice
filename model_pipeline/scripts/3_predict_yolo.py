from ultralytics import YOLO
from pathlib import Path

# paths
MODEL_PATH = Path("runs/detect/runs/football_ball_yolo4/weights/best.pt")
SOURCE = Path("datasets/test/images")

# load model
model = YOLO(MODEL_PATH)

# run inference
results = model.predict(
    source=SOURCE,
    conf=0.25,
    imgsz=640,
    device="cpu",
    save=True,
)

print("Inference finished.")

