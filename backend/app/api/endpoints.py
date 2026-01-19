try:
	from fastapi import FastAPI, UploadFile, File, Form  # type: ignore
	from fastapi.responses import JSONResponse  # type: ignore
except Exception as exc:
	raise ImportError(
		"FastAPI is required but not installed. Install it with 'pip install fastapi uvicorn'."
	) from exc
from pathlib import Path
from typing import List
import shutil

from app.services.ingestion import ingest_documents
from app.data.vector_store import VectorStore
from app.services.qa import generate_answer

app = FastAPI(title="DocQA Proto")

@app.post("/ingest")
async def ingest_endpoint(files: List[UploadFile] = File(...)):
    # Save uploaded files to temp paths inside ./uploads
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

    store = VectorStore()
    results = ingest_documents([str(p) for p in temp_paths], store=store)
    docs = [
        {"doc_id": res.doc_id, "num_chunks": res.num_chunks, "filename": name}
        for res, name in zip(results, filenames)
    ]
    return {"ok": True, "documents": docs}

@app.post("/ask")
async def ask_endpoint(question: str = Form(...), k: int = Form(8), min_score: float = Form(0.4)):
    ans = generate_answer(question=question, k=k, min_score=min_score)
    # Format proofs for response (short text)
    proofs = [{
        "doc_id": p.doc_id, "page": p.page, "score": p.score,
        "start": p.start, "end": p.end, "text": p.text.strip()
    } for p in ans.proofs]
    return JSONResponse({"ok": True, "answer": ans.answer, "proofs": proofs})
