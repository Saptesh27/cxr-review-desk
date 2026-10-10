"""Agent 4. CheXagent reads the film together with the typed clinical notes."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import torch
from PIL import Image

from cxr.agents.base import AgentResult, blank_votes, failed
from cxr.labels import LABELS

MODEL_ID = "StanfordAIMI/CheXagent-2-3b"
SKIPPED = (
    "This computer has no NVIDIA GPU, so the notes model stays off. "
    "The other three agents still run."
)


class MllmAgent:
    key = "mllm"
    name = "Clinical notes"
    algorithm = "CheXagent-2-3B"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._error: str | None = None
        self._loaded = False
        self._skipped = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not torch.cuda.is_available():
            self._skipped = True
            self._error = SKIPPED
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                trust_remote_code=True,
            )
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = model.to(dtype=dtype)
            model.eval()
            self._model = model
        except Exception as exc:
            self._error = f"Notes model could not load: {exc}"

    def predict(self, image: Image.Image, notes: str) -> AgentResult:
        self.load()
        if self._skipped:
            return failed(self.key, self.name, self.algorithm, SKIPPED, status="skipped")
        if self._error or self._model is None:
            return failed(self.key, self.name, self.algorithm, self._error or "Model unavailable")
        path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                image.convert("RGB").save(handle.name, format="JPEG", quality=90)
                path = handle.name
            prompt = _prompt(notes)
            query = self._tokenizer.from_list_format([{"image": path}, {"text": prompt}])
            conversation = [
                {"from": "system", "value": "You are a careful chest X-ray assistant."},
                {"from": "human", "value": query},
            ]
            input_ids = self._tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            with torch.no_grad():
                output = self._model.generate(
                    input_ids.to(device),
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                    use_cache=True,
                    max_new_tokens=400,
                )[0]
            text = self._tokenizer.decode(output[input_ids.shape[-1] :], skip_special_tokens=True)
            votes = _parse_votes(text)
            mentioned = [label for label, score in votes.items() if score is not None]
            summary = "No listed finding in the reply." if not mentioned else "Noted: " + ", ".join(mentioned)
            return AgentResult(
                key=self.key,
                name=self.name,
                algorithm=self.algorithm,
                status="ready",
                probabilities=votes,
                summary=summary,
                detail=text.strip(),
            )
        except Exception as exc:
            return failed(self.key, self.name, self.algorithm, f"Notes model failed: {exc}")
        finally:
            if path:
                Path(path).unlink(missing_ok=True)


def _prompt(notes: str) -> str:
    clinical = notes.strip() or "No clinical notes were provided."
    names = ", ".join(LABELS)
    return (
        f"Clinical notes: {clinical}\n"
        f"Use only these finding names: {names}.\n"
        "Return a JSON list only. Each item has keys label and confidence. "
        "confidence is a number from 0 to 1. Include a finding only when you believe it is present. "
        "If none are present, return []."
    )


def _parse_votes(text: str) -> dict[str, float | None]:
    votes = blank_votes()
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    parsed = None
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
    lookup = {label.lower(): label for label in LABELS}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            raw_label = str(item.get("label", "")).strip().lower()
            label = lookup.get(raw_label)
            if label is None:
                continue
            try:
                confidence = float(item.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0.6
            votes[label] = max(0.0, min(1.0, confidence))
        return votes
    lowered = cleaned.lower()
    for label in LABELS:
        if label.lower() in lowered:
            votes[label] = 0.6
    return votes
