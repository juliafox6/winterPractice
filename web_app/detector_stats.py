import json
from datetime import datetime
from pathlib import Path
from typing import List
from pydantic import BaseModel

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DETECT_HISTORY = DATA_DIR / "detections.json"


class DetectionEntry(BaseModel):
    id: str
    timestamp: datetime
    filename: str
    objects_count: int
    avg_confidence: float
    result_url: str


def load_detections() -> List[DetectionEntry]:
    if not DETECT_HISTORY.exists():
        return []
    raw = json.loads(DETECT_HISTORY.read_text(encoding="utf-8"))
    return [DetectionEntry(**x) for x in raw]


def save_detection(entry: DetectionEntry):
    history = load_detections()
    history.append(entry)
    DETECT_HISTORY.write_text(
        json.dumps([e.dict() for e in history], indent=2, default=str),
        encoding="utf-8",
    )

