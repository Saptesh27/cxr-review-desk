"""Agent 3. YOLOv8-World proposes boxes for the shared finding names."""

from __future__ import annotations

from PIL import Image

from cxr.agents.base import AgentResult, blank_votes, failed, top_summary
from cxr.heatmaps import draw_boxes, to_png
from cxr.labels import LABELS
from cxr.paths import CHECKPOINTS, ensure_dirs


class DetectorAgent:
    key = "detector"
    name = "Localization"
    algorithm = "YOLOv8-World"

    def __init__(self) -> None:
        self._model = None
        self._error: str | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            ensure_dirs()
            from ultralytics import YOLO

            weight = CHECKPOINTS / "yolov8s-worldv2.pt"
            source = str(weight) if weight.exists() else "yolov8s-worldv2.pt"
            model = YOLO(source)
            model.set_classes(LABELS)
            self._model = model
            if not weight.exists():
                downloaded = CHECKPOINTS.parent / "yolov8s-worldv2.pt"
                if downloaded.exists():
                    downloaded.replace(weight)
        except Exception as exc:
            self._error = f"Localization model could not load: {exc}"

    def predict(self, image: Image.Image) -> AgentResult:
        self.load()
        if self._error or self._model is None:
            return failed(self.key, self.name, self.algorithm, self._error or "Model unavailable")
        try:
            device = "0" if _cuda() else "cpu"
            results = self._model.predict(
                source=image.convert("RGB"),
                conf=0.08,
                verbose=False,
                device=device,
            )
            votes = blank_votes()
            for label in LABELS:
                votes[label] = 0.0
            boxes = []
            result = results[0]
            names = result.names or {}
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if isinstance(names, dict):
                        label = str(names.get(class_id, names.get(str(class_id), "")))
                    else:
                        label = str(names[class_id])
                    confidence = float(box.conf.item())
                    if label not in votes:
                        continue
                    votes[label] = max(votes[label] or 0.0, confidence)
                    xyxy = [float(value) for value in box.xyxy[0].tolist()]
                    boxes.append({"label": label, "confidence": confidence, "xyxy": xyxy})
            boxes.sort(key=lambda item: item["confidence"], reverse=True)
            drawn = draw_boxes(image, [box for box in boxes if box["confidence"] >= 0.15])
            found = [box["label"] for box in boxes[:4]]
            summary = "No box above the display threshold." if not found else "Boxes: " + ", ".join(found)
            if any((votes[label] or 0) >= 0.2 for label in LABELS):
                summary = top_summary(votes)
            return AgentResult(
                key=self.key,
                name=self.name,
                algorithm=self.algorithm,
                status="ready",
                probabilities=votes,
                summary=summary,
                detail="Anchor-free boxes from the text prompts of the 12 findings.",
                heatmap_png=to_png(drawn),
            )
        except Exception as exc:
            return failed(self.key, self.name, self.algorithm, f"Localization model failed: {exc}")


def _cuda() -> bool:
    import torch

    return bool(torch.cuda.is_available())
