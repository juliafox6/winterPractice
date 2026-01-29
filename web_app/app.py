import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from starlette.templating import Jinja2Templates
import subprocess

# --------------------
# Paths
# --------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
MODEL_PATH = (
    BASE_DIR.parent
    / "model_pipeline"
    / "runs"
    / "detect"
    / "runs"
    / "football_ball_yolo4"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# App
# --------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------
# Model
# --------------------
model = YOLO(str(MODEL_PATH))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    suffix = upload_path.suffix.lower()

    # ---------------- IMAGE ----------------
    if suffix in [".jpg", ".jpeg", ".png"]:
        result_dir = RESULTS_DIR / file_id
        result_dir.mkdir(exist_ok=True)

        model.predict(
            source=str(upload_path),
            save=True,
            conf=0.25,
            imgsz=640,
            device="cpu",
            project=str(RESULTS_DIR),
            name=file_id,
            exist_ok=True,
        )

        output_image = next(result_dir.glob("*.jpg"))

        return JSONResponse({
            "type": "image",
            "result_url": f"/static/results/{file_id}/{output_image.name}",
            "save_path": str(result_dir),
        })

    # ---------------- VIDEO ----------------
    elif suffix in [".mp4", ".avi", ".mov"]:
        result_dir = RESULTS_DIR / file_id
        result_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(upload_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        output_video = result_dir / "result.mp4"

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (width, height),
        )
       
        if not writer.isOpened():
            print("avc1 is not supported, falling back to mp4v")
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        if not writer.isOpened():
            cap.release()
            return JSONResponse(
                {"error": "Could not initialize video writer (codec issue)"},
                status_code=500,
            )

        frame_count = 0

        for r in model.predict(
            source=str(upload_path),
            stream=True,
            conf=0.25,
            imgsz=640,
            device="cpu",
        ):
            frame = r.plot()
            writer.write(frame)
            frame_count += 1

        cap.release()
        writer.release()
        
        final_video = result_dir / "result_browser.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(output_video),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            str(final_video),
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return JSONResponse({
            "type": "video",
            "result_url": f"/static/results/{file_id}/result_browser.mp4",
            "save_path": str(result_dir),
            "frames_processed": frame_count,
        })

    else:
        return JSONResponse(
            {"error": "Unsupported file format"},
            status_code=400
        )

