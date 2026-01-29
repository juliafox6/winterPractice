import shutil
import uuid
from pathlib import Path
from datetime import datetime
import subprocess
import time

import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from starlette.templating import Jinja2Templates

from stats import router, RequestEntry, save_entry, load_history, get_summary, generate_pdf, generate_excel
from detector_stats import save_detection, load_detections, DetectionEntry

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
app.include_router(router)

# --------------------
# Model
# --------------------
model = YOLO(str(MODEL_PATH))

# --------------------
# Routes
# --------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    file_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    suffix = upload_path.suffix.lower()
    input_data = {"filename": file.filename, "type": "image" if suffix in [".jpg", ".jpeg", ".png"] else "video"}

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

        elapsed = (time.time() - start_time) * 1000  # ms
        save_entry(RequestEntry(
            timestamp=datetime.utcnow(),
            endpoint="/predict",
            payload=input_data,
            processing_time_ms=elapsed,
        ))

        detections = 0
        avg_conf = 0.0

        for r in model.predict(
            source=str(upload_path),
            conf=0.25,
            imgsz=640,
            device="cpu",
        ):
            if r.boxes is not None and len(r.boxes) > 0:
                detections = len(r.boxes)
                avg_conf = float(r.boxes.conf.mean())

        save_detection(DetectionEntry(
            id=file_id[:8],
            timestamp=datetime.utcnow(),
            filename=file.filename,
            objects_count=detections,
            avg_confidence=round(avg_conf, 2),
            result_url=f"/static/results/{file_id}/{output_image.name}",
        ))

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

        temp_video = result_dir / "temp.mp4"
        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            return JSONResponse({"error": "Could not initialize video writer"}, status_code=500)

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

        # Конвертируем в совместимый mp4 для браузера
        final_video = result_dir / "result_browser.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(temp_video),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            str(final_video)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_video.unlink(missing_ok=True)

        elapsed = (time.time() - start_time) * 1000
        save_entry(RequestEntry(
            timestamp=datetime.utcnow(),
            endpoint="/predict",
            payload=input_data,
            processing_time_ms=elapsed,
        ))

        return JSONResponse({
            "type": "video",
            "result_url": f"/static/results/{file_id}/result_browser.mp4",
            "save_path": str(result_dir),
            "frames_processed": frame_count,
        })

    else:
        return JSONResponse({"error": "Unsupported file format"}, status_code=400)


# --------------------
# Stats Endpoints
# --------------------
@app.get("/stats/history")
def stats_history():
    return load_history()


@app.get("/stats/summary")
def stats_summary():
    return get_summary()


@app.get("/stats/report/pdf")
def stats_pdf():
    path = generate_pdf()
    return FileResponse(path, media_type="application/pdf", filename="report.pdf")


@app.get("/stats/report/excel")
def stats_excel():
    path = generate_excel()
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="report.xlsx"
    )

@app.get("/detections", response_class=HTMLResponse)
def detections_page(request: Request):
    history = load_detections()
    return templates.TemplateResponse(
        "detections.html",
        {"request": request, "history": history}
    )

