from ultralytics import YOLO
from pathlib import Path

# -----------------------------
# Пути
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "datasets" / "data.yaml"

# -----------------------------
# Конфиг
# -----------------------------
MODEL_NAME = "yolov8n.pt"   # предобученная лёгкая модель
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16

# -----------------------------
# Загрузка модели
# -----------------------------
print("Loading pre-trained YOLO model...")
model = YOLO(MODEL_NAME)

# -----------------------------
# Тренировка
# -----------------------------
print("Starting training...")
results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    project="runs",
    name="football_ball_yolo",
    pretrained=True,
    device="cpu"
)

print("Training finished.")

# -----------------------------
# Сохранение лучшей модели
# -----------------------------
best_model_path = Path("runs/detect/football_ball_yolo/weights/best.pt")

if best_model_path.exists():
    print(f"Best model saved at: {best_model_path}")
else:
    print("Warning: best.pt not found!")

