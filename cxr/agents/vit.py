"""Agent 2. A ViT-B/16 compares the whole film with present and absent prompts."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from cxr.agents.base import AgentResult, failed, top_summary
from cxr.heatmaps import overlay, to_png
from cxr.labels import LABELS

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _as_grid(values: np.ndarray) -> np.ndarray | None:
    side = int(np.sqrt(values.size))
    if side < 2:
        return None
    if side * side != values.size:
        values = values[-side * side :]
    return values.reshape(side, side)


class VitAgent:
    key = "vit"
    name = "Global pattern"
    algorithm = "BiomedCLIP ViT-B/16"

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None
        self._error: str | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import open_clip

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, preprocess = open_clip.create_model_from_pretrained(MODEL_ID)
            self._tokenizer = open_clip.get_tokenizer(MODEL_ID)
            model.eval()
            self._model = model.to(self._device)
            self._preprocess = preprocess
        except Exception as exc:
            self._error = f"Global pattern model could not load: {exc}"

    def _attention_map(self, image_tensor: torch.Tensor) -> np.ndarray | None:
        trunk = getattr(self._model.visual, "trunk", None)
        blocks = getattr(trunk, "blocks", None)
        if blocks:
            return self._timm_cls_attention(image_tensor, blocks[-1].attn, trunk)
        try:
            return self._open_clip_rollout(image_tensor, self._model.visual.transformer.resblocks)
        except AttributeError:
            return None

    def _timm_cls_attention(self, image_tensor: torch.Tensor, attention, trunk) -> np.ndarray | None:
        captured: list[torch.Tensor] = []

        def hook(_module, inputs, _output):
            tokens = inputs[0]
            batch, count, _channels = tokens.shape
            qkv = attention.qkv(tokens)
            qkv = qkv.reshape(batch, count, 3, attention.num_heads, attention.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, _value = qkv.unbind(0)
            query, key = attention.q_norm(query), attention.k_norm(key)
            weights = (query * attention.scale) @ key.transpose(-2, -1)
            weights = weights.softmax(dim=-1).mean(dim=1)
            captured.append(weights.detach())

        handle = attention.register_forward_hook(hook)
        try:
            with torch.no_grad():
                self._model.encode_image(image_tensor)
        except Exception:
            return None
        finally:
            handle.remove()
        if not captured:
            return None
        prefix = int(getattr(trunk, "num_prefix_tokens", 1))
        cls_to_patches = captured[0][0, 0, prefix:].detach().cpu().numpy()
        return _as_grid(cls_to_patches)

    def _open_clip_rollout(self, image_tensor: torch.Tensor, blocks) -> np.ndarray | None:
        captured: list[torch.Tensor] = []
        originals = []
        for block in blocks:
            originals.append(block.attention)

            def _wrapped(q_x, _block=block):
                attn_mask = _block.attn_mask
                if attn_mask is not None:
                    attn_mask = attn_mask.to(dtype=q_x.dtype, device=q_x.device)
                output, weights = _block.attn(
                    q_x,
                    q_x,
                    q_x,
                    need_weights=True,
                    average_attn_weights=True,
                    attn_mask=attn_mask,
                )
                captured.append(weights.detach())
                return output

            block.attention = _wrapped
        try:
            with torch.no_grad():
                self._model.encode_image(image_tensor)
        except Exception:
            return None
        finally:
            for block, original in zip(blocks, originals):
                block.attention = original
        if not captured:
            return None
        token_count = captured[0].shape[-1]
        rollout = torch.eye(token_count, device=captured[0].device)
        for weights in captured:
            matrix = weights[0] + torch.eye(token_count, device=weights.device)
            matrix = matrix / matrix.sum(dim=-1, keepdim=True)
            rollout = matrix @ rollout
        return _as_grid(rollout[0, 1:].detach().cpu().numpy())

    def predict(self, image: Image.Image) -> AgentResult:
        self.load()
        if self._error or self._model is None:
            return failed(self.key, self.name, self.algorithm, self._error or "Model unavailable")
        try:
            prompts = []
            for label in LABELS:
                name = label.lower()
                prompts.append(f"chest radiograph showing {name}")
                prompts.append(f"chest radiograph with no {name}")
            image_tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
            text = self._tokenizer(prompts).to(self._device)
            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(text)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scale = self._model.logit_scale.exp()
                logits = scale * image_features @ text_features.T
            votes: dict[str, float | None] = {}
            best_label = LABELS[0]
            best_score = -1.0
            for index, label in enumerate(LABELS):
                pair = logits[0, index * 2 : index * 2 + 2]
                score = float(torch.softmax(pair, dim=0)[0].item())
                votes[label] = score
                if score > best_score:
                    best_score = score
                    best_label = label
            heat = self._attention_map(image_tensor)
            heatmap = to_png(overlay(image, heat)) if heat is not None else None
            if heat is not None:
                detail = (
                    f"Class-token attention on the last transformer block. "
                    f"The strongest present-prompt was {best_label}."
                )
            else:
                detail = f"The strongest present-prompt was {best_label}."
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
            return failed(self.key, self.name, self.algorithm, f"Global pattern model failed: {exc}")
