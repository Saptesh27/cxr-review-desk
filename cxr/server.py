"""Local review desk. Open http://127.0.0.1:7860 after the server starts."""

from __future__ import annotations

from cxr.console import configure_stdio

configure_stdio()

import io
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from cxr.db import delete_study, get_study, init_db, latest_training, list_studies
from cxr.fusion import load_fusion, per_label_f1
from cxr.paths import FUSION_PATH, NIH_SMALL, WEB, ensure_dirs
from cxr.pipeline import REGISTRY, remove_files, run_study

app = FastAPI(title="CXR Review Desk")
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
BUSY = threading.Event()
MAX_BYTES = 20 * 1024 * 1024


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()
    init_db()


TEMPLATES = WEB / "templates"
PAGES = {
    "new": {"path": "/", "title": "New study", "file": "new.html", "scripts": ["/assets/new.js"]},
    "history": {"path": "/history", "title": "History", "file": "history.html", "scripts": ["/assets/history.js"]},
    "training": {"path": "/training", "title": "Training record", "file": "training.html", "scripts": ["/assets/training.js"]},
    "about": {"path": "/about", "title": "How it works", "file": "about.html", "scripts": []},
}


def render_page(key: str) -> HTMLResponse:
    spec = PAGES[key]
    layout = (TEMPLATES / "layout.html").read_text(encoding="utf-8")
    body = (TEMPLATES / spec["file"]).read_text(encoding="utf-8")
    scripts = "\n".join(f'<script src="{src}"></script>' for src in spec["scripts"])
    html = (
        layout.replace("{{title}}", spec["title"])
        .replace("{{heading}}", spec["title"])
        .replace("{{page}}", key)
        .replace("{{body}}", body)
        .replace("{{scripts}}", scripts)
    )
    html = html.replace(f'data-nav="{key}"', f'data-nav="{key}" aria-current="page"')
    return HTMLResponse(html)


@app.get("/")
def new_study_page() -> HTMLResponse:
    return render_page("new")


@app.get("/history")
def history_page() -> HTMLResponse:
    return render_page("history")


@app.get("/training")
def training_page() -> HTMLResponse:
    return render_page("training")


@app.get("/about")
def about_page() -> HTMLResponse:
    return render_page("about")


app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/api/status")
def status() -> dict:
    gpu = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if gpu else ""
    return {
        "cuda": gpu,
        "gpu_name": name,
        "fusion_trained": FUSION_PATH.exists(),
        "busy": BUSY.is_set(),
    }


@app.get("/api/training")
def training() -> dict:
    return {"run": latest_training(), "labels": _label_rows()}


def _label_rows() -> list[dict]:
    cache = NIH_SMALL / "features.npz"
    payload = load_fusion()
    if payload is None or not cache.exists():
        return []
    stored = np.load(cache, allow_pickle=False)
    splits = np.asarray(stored["splits"]).astype(str)
    test = splits == "test"
    train = splits == "train"
    rows = per_label_f1(payload, stored["features"][test], stored["targets"][test])
    train_hits = np.asarray(stored["targets"][train]).sum(axis=0)
    for index, row in enumerate(rows):
        row["train_positives"] = int(train_hits[index])
    return rows


@app.get("/api/studies")
def studies() -> dict:
    return {"studies": list_studies()}


@app.get("/api/studies/{study_id}")
def study(study_id: int) -> dict:
    payload = get_study(study_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Study not found.")
    return payload


@app.delete("/api/studies/{study_id}")
def remove_study(study_id: int) -> dict:
    removed = delete_study(study_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="Study not found.")
    remove_files(removed["paths"])
    return {"ok": True}


@app.get("/api/studies/{study_id}/image")
def study_image(study_id: int) -> FileResponse:
    payload = get_study(study_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Study not found.")
    path = _image_path(study_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file is missing.")
    return FileResponse(path)


@app.get("/api/studies/{study_id}/heatmap/{agent}")
def study_heatmap(study_id: int, agent: str) -> FileResponse:
    if agent not in {"cnn", "vit", "detector"}:
        raise HTTPException(status_code=404, detail="No map for that agent.")
    path = _image_path(study_id).parent / f"{agent}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Map not found.")
    return FileResponse(path)


@app.post("/api/studies")
async def create_study(image: UploadFile = File(...), notes: str = Form("")) -> dict:
    raw = await image.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 20 MB.")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="That file is not an image.") from exc
    if BUSY.is_set():
        raise HTTPException(status_code=409, detail="A study is already running.")
    BUSY.set()
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"state": "running", "step": "Queued", "index": 0, "total": 6}
        if len(JOBS) > 40:
            for old in list(JOBS)[:-40]:
                JOBS.pop(old, None)
    thread = threading.Thread(target=_run_job, args=(job_id, pil.copy(), notes.strip()), daemon=True)
    try:
        thread.start()
    except Exception:
        BUSY.clear()
        raise
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    with JOBS_LOCK:
        payload = JOBS.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return payload


def _run_job(job_id: str, image: Image.Image, notes: str) -> None:
    def progress(step: str, index: int, total: int) -> None:
        with JOBS_LOCK:
            JOBS[job_id] = {"state": "running", "step": step, "index": index, "total": total}

    try:
        study = run_study(image, notes, progress)
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id] = {"state": "error", "message": str(exc)}
        return
    finally:
        BUSY.clear()
    with JOBS_LOCK:
        JOBS[job_id] = {"state": "done", "study": study}


def _image_path(study_id: int) -> Path:
    from cxr.db import connect

    with connect() as connection:
        row = connection.execute("SELECT image_path FROM studies WHERE id = ?", (study_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Study not found.")
    return Path(row["image_path"])
