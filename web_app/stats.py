# stats.py
# Модуль статистики и отчётов

import json
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from reportlab.platypus import SimpleDocTemplate, Paragraph, Table
from reportlab.lib.styles import getSampleStyleSheet
from openpyxl import Workbook

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.json"

router = APIRouter(prefix="/stats", tags=["stats"])


# --------- МОДЕЛИ ---------
class RequestEntry(BaseModel):
    timestamp: datetime
    endpoint: str
    payload: dict
    processing_time_ms: float


# --------- ХРАНЕНИЕ ---------
def load_history() -> List[dict]:
    if not HISTORY_FILE.exists():
        return []
    return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))


def save_entry(entry: RequestEntry):
    history = load_history()
    history.append(entry.dict())
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# --------- API ---------
@router.get("/history")
def get_history():
    return load_history()


@router.get("/summary")
def get_summary():
    history = load_history()
    if not history:
        return {"total": 0}

    total = len(history)
    avg_time = sum(e["processing_time_ms"] for e in history) / total
    by_endpoint = {}
    for e in history:
        by_endpoint[e["endpoint"]] = by_endpoint.get(e["endpoint"], 0) + 1

    return {
        "total_requests": total,
        "average_processing_ms": round(avg_time, 2),
        "by_endpoint": by_endpoint,
    }


# --------- ОТЧЁТЫ ---------
def generate_pdf(path: Path):
    history = load_history()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path))

    elements = [Paragraph("Отчёт по запросам", styles["Title"])]

    table_data = [["Время", "Endpoint", "Время обработки (мс)"]]
    for e in history:
        table_data.append([
            e["timestamp"],
            e["endpoint"],
            str(e["processing_time_ms"]),
        ])

    elements.append(Table(table_data))
    doc.build(elements)



def generate_excel(path: Path):
    history = load_history()
    wb = Workbook()
    ws = wb.active
    ws.title = "Requests"

    ws.append(["Время", "Endpoint", "Payload", "Время обработки (мс)"])
    for e in history:
        ws.append([
            e["timestamp"],
            e["endpoint"],
            json.dumps(e["payload"], ensure_ascii=False),
            e["processing_time_ms"],
        ])

    wb.save(path)


@router.get("/report/pdf")
def pdf_report():
    path = DATA_DIR / "report.pdf"
    generate_pdf(path)
    return {"file": str(path)}


@router.get("/report/excel")
def excel_report():
    path = DATA_DIR / "report.xlsx"
    generate_excel(path)
    return {"file": str(path)}

