"""Run the four agents, fuse them, and store one study."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from cxr.agents.base import AgentResult
from cxr.agents.cnn import CnnAgent
from cxr.agents.detector import DetectorAgent
from cxr.agents.mllm import MllmAgent
from cxr.agents.vit import VitAgent
from cxr.db import get_study, insert_study, save_decision, save_report
from cxr.fusion import combine
from cxr.paths import UPLOADS, ensure_dirs


class Registry:
    def __init__(self) -> None:
        self.cnn = CnnAgent()
        self.vit = VitAgent()
        self.detector = DetectorAgent()
        self.mllm = MllmAgent()
        self.lock = threading.Lock()


REGISTRY = Registry()
Progress = Callable[[str, int, int], None]


def run_study(image: Image.Image, notes: str, progress: Progress | None = None) -> dict:
    ensure_dirs()
    total = 6

    def step(message: str, index: int) -> None:
        if progress:
            progress(message, index, total)

    with REGISTRY.lock:
        step("Reading the film", 1)
        step("Local texture model", 2)
        cnn = REGISTRY.cnn.predict(image)
        step("Global pattern model", 3)
        vit = REGISTRY.vit.predict(image)
        step("Localization model", 4)
        detector = REGISTRY.detector.predict(image)
        step("Clinical notes model", 5)
        notes_agent = REGISTRY.mllm.predict(image, notes)
        step("Saving the study", 6)
        decision = combine(
            {
                "cnn": cnn.probabilities,
                "vit": vit.probabilities,
                "detector": detector.probabilities,
                "mllm": notes_agent.probabilities,
            }
        )
        study_id = _store(image, notes, [cnn, vit, detector, notes_agent], decision)
    study = get_study(study_id)
    if study is None:
        raise RuntimeError("The study was saved but could not be read back.")
    return study


def _store(image: Image.Image, notes: str, agents: list[AgentResult], decision: dict) -> int:
    study_id = insert_study("", notes)
    folder = UPLOADS / str(study_id)
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / "original.jpg"
    image.convert("RGB").save(image_path, format="JPEG", quality=90)
    from cxr.db import connect

    with connect() as connection:
        connection.execute("UPDATE studies SET image_path = ? WHERE id = ?", (str(image_path), study_id))
    for agent in agents:
        heatmap_path = None
        if agent.heatmap_png:
            heatmap_file = folder / f"{agent.key}.png"
            heatmap_file.write_bytes(agent.heatmap_png)
            heatmap_path = str(heatmap_file)
        report = agent.as_dict(None)
        report["heatmap_path"] = heatmap_path
        save_report(study_id, report)
    save_decision(study_id, decision)
    return study_id


def embed_image_agents(image: Image.Image) -> dict[str, dict[str, float | None]]:
    """Used by fusion training. The notes model is not run."""
    with REGISTRY.lock:
        agents = [
            REGISTRY.cnn.predict(image),
            REGISTRY.vit.predict(image),
            REGISTRY.detector.predict(image),
        ]
    failed_agents = [agent.name for agent in agents if agent.status == "error"]
    if failed_agents:
        detail = "; ".join(agent.error or agent.summary for agent in agents if agent.status == "error")
        raise RuntimeError(f"Training stopped because {', '.join(failed_agents)} failed. {detail}")
    return {agent.key: agent.probabilities for agent in agents}


def remove_files(paths: list[str]) -> None:
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            path.unlink(missing_ok=True)
        parent = path.parent
        if parent.exists() and parent != UPLOADS and not any(parent.iterdir()):
            parent.rmdir()
