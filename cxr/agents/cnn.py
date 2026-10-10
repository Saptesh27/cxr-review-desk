"""Agent 1. DenseNet-121 reads local texture and explains it with Grad-CAM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cxr.agents.base import AgentResult, failed, top_summary
from cxr.heatmaps import overlay, to_png
from cxr.labels import LABELS, XRV_NAME


class CnnAgent:
    key = "cnn"
    name = "Local texture"
    algorithm = "DenseNet-121"

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._pathologies: list[str] = []
        self._error: str | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from cxr.console import configure_stdio

            configure_stdio()
            import torchxrayvision as xrv
            import torchvision

            self._xrv = xrv
            self._transform = torchvision.transforms.Compose(
                [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
            )
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = xrv.models.DenseNet(weights="densenet121-res224-all", apply_sigmoid=True)
            model.eval()
            self._model = model.to(self._device)
            self._pathologies = list(xrv.datasets.default_pathologies)
        except Exception as exc:
            self._error = f"Local texture model could not load: {exc}"

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        gray = np.array(image.convert("L"))
        normalized = self._xrv.datasets.normalize(gray, 255)
        sample = normalized[None, :, :]
        sample = self._transform(sample)
        tensor = torch.from_numpy(sample).unsqueeze(0)
        return tensor.to(self._device).float()

    def _grad_cam(self, tensor: torch.Tensor, class_index: int) -> np.ndarray:
        activations: list[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            cloned = output.clone()
            cloned.retain_grad()
            activations.append(cloned)
            return cloned

        handle = self._model.features.norm5.register_forward_hook(forward_hook)
        try:
            self._model.zero_grad(set_to_none=True)
            output = self._model(tensor.detach())
            output[0, class_index].backward()
            activation = activations[0]
            gradient = activation.grad
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1))[0]
            cam = cam - cam.min()
            peak = cam.max()
            if float(peak) > 0:
                cam = cam / peak
            return cam.detach().cpu().numpy()
        finally:
            handle.remove()

    def predict(self, image: Image.Image) -> AgentResult:
        self.load()
        if self._error or self._model is None:
            return failed(self.key, self.name, self.algorithm, self._error or "Model unavailable")
        try:
            tensor = self._tensor(image)
            with torch.no_grad():
                output = self._model(tensor)[0].detach().cpu()
            votes: dict[str, float | None] = {}
            best_label = None
            best_score = -1.0
            best_index = None
            for label in LABELS:
                pathology = XRV_NAME[label]
                if pathology not in self._pathologies:
                    votes[label] = None
                    continue
                index = self._pathologies.index(pathology)
                score = float(np.clip(output[index].item(), 0.0, 1.0))
                votes[label] = score
                if score > best_score:
                    best_score = score
                    best_label = label
                    best_index = index
            heatmap = None
            detail = ""
            if best_index is not None:
                try:
                    heat = self._grad_cam(tensor, best_index)
                    heatmap = to_png(overlay(image, heat))
                    detail = f"Grad-CAM on the last dense block for {best_label}."
                except Exception as exc:
                    detail = f"Scores are ready. The map could not be drawn: {exc}"
            return AgentResult(
                key=self.key,
                name=self.name,
                algorithm=self.algorithm,
                status="ready",
                probabilities=votes,
                summary=top_summary(votes),
                detail=detail,
                heatmap_png=heatmap,
            )
        except Exception as exc:
            return failed(self.key, self.name, self.algorithm, f"Local texture model failed: {exc}")
