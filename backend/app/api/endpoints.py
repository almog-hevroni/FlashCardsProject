try:
    from fastapi import FastAPI, UploadFile, File, Form, Header  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
except Exception as exc:
    raise ImportError(
        "FastAPI is required but not installed. Install it with 'pip install fastapi uvicorn'."
    ) from exc
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
from datetime import datetime, timezone

from app.services.ingestion import ingest_documents
from app.data.vector_store import VectorStore
from app.services.qa import generate_answer
from app.services.exams import attach_documents, ImmutableExamError
from app.services.diagnostic_lifecycle import (
    DiagnosticBootstrapError,
    DiagnosticLifecycleService,
)
from app.services.diagnostic_review_reducer import DiagnosticReviewReducer

app = FastAPI(title="DocQA Proto")


def _immutable_exam_error_payload(*, exam_id: Optional[str] = None, message: str) -> Dict[str, Any]:
    return {
        "error": {
            "code": "immutable_exam",
            "message": message,
            "exam_id": exam_id,
        }
    }


def _extract_uploaded_files(files: List[UploadFile]) -> tuple[List[Path], List[str]]:
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    temp_paths: List[Path] = []
    filenames: List[str] = []
    for file in files:
        temp_path = uploads / file.filename
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        temp_paths.append(temp_path)
        filenames.append(file.filename)
    return temp_paths, filenames

@app.post("/exams/from-upload")
async def create_exam_from_upload_endpoint(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    title: str = Form(...),
    mode: str = Form(default="mastery"),
):
    temp_paths, filenames = _extract_uploaded_files(files)
    store = VectorStore()
    svc = DiagnosticLifecycleService(store=store)
    try:
        result = svc.bootstrap_exam_from_upload(
            user_id=user_id,
            title=title,
            mode=mode,
            paths=[str(p) for p in temp_paths],
            info={"source": "from_upload", "filenames": filenames},
        )
    except DiagnosticBootstrapError as exc:
        return JSONResponse(
            {"error": "diagnostic_bootstrap_failed", "message": str(exc)},
            status_code=422,
        )
    except ImmutableExamError as exc:
        return JSONResponse(
            _immutable_exam_error_payload(exam_id=exc.exam_id, message=str(exc)),
            status_code=409,
        )
    return {
        "exam_id": result.exam_id,
        "state": result.state,
        "diagnostic_total": result.diagnostic_total,
        "diagnostic_answered": result.diagnostic_answered,
        "cards_generated": result.cards_generated,
        "topic_count": result.topic_count,
    }


@app.get("/exams/{exam_id}")
async def get_exam_endpoint(exam_id: str, user_id: str):
    store = VectorStore()
    exam = store.db.get_exam(exam_id)
    if exam is None:
        return JSONResponse({"error": f"Exam not found: {exam_id}"}, status_code=404)
    if exam.user_id != user_id:
        return JSONResponse({"error": "Exam does not belong to user"}, status_code=403)
    return {
        "exam_id": exam.exam_id,
        "user_id": exam.user_id,
        "title": exam.title,
        "mode": exam.mode,
        "state": exam.state,
        "diagnostic_total": exam.diagnostic_total,
        "diagnostic_answered": exam.diagnostic_answered,
        "diagnostic_started_at": exam.diagnostic_started_at,
        "diagnostic_completed_at": exam.diagnostic_completed_at,
        "created_at": exam.created_at,
        "updated_at": exam.updated_at,
        "info": exam.info,
    }


@app.post("/exams/{exam_id}/cards/{card_id}/review")
async def review_card_endpoint(
    exam_id: str,
    card_id: str,
    user_id: str = Form(...),
    rating: str = Form(...),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    key = (idempotency_key or "").strip() or f"{user_id}:{exam_id}:{card_id}:{rating}:{datetime.now(timezone.utc).isoformat()}"
    reducer = DiagnosticReviewReducer()
    try:
        result = reducer.apply_review(
            user_id=user_id,
            exam_id=exam_id,
            card_id=card_id,
            rating=rating,
            idempotency_key=key,
        )
    except ValueError as exc:
        message = str(exc)
        status = 400
        if "not found" in message.lower():
            status = 404
        if "does not belong" in message.lower():
            status = 403
        return JSONResponse({"error": message}, status_code=status)

    return {
        "review_id": result.review_id,
        "card_id": result.card_id,
        "rating": result.rating,
        "topic_proficiency": result.topic_proficiency,
        "diagnostic_answered": result.diagnostic_answered,
        "diagnostic_total": result.diagnostic_total,
        "exam_state": result.exam_state,
        "idempotent_replay": result.idempotent_replay,
    }
