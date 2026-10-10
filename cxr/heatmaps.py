"""Turn a score map or a set of boxes into a PNG the review desk can show."""

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _resize_heat(heat: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    heat_image = Image.fromarray(np.clip(heat * 255, 0, 255).astype(np.uint8))
    heat_image = heat_image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(heat_image).astype(np.float32) / 255.0


def overlay(base: Image.Image, heat: np.ndarray) -> Image.Image:
    """Amber overlay. Low scores stay transparent so the film stays readable."""
    rgb = base.convert("RGB")
    width, height = rgb.size
    heat = _resize_heat(np.asarray(heat, dtype=np.float32), (width, height))
    heat = heat - float(heat.min())
    peak = float(heat.max())
    if peak > 1e-6:
        heat = heat / peak
    alpha = np.clip((heat - 0.28) / 0.72, 0, 1)[..., None] * 0.72
    color = np.zeros((*heat.shape, 3), dtype=np.float32)
    color[..., 0] = 214
    color[..., 1] = 122
    color[..., 2] = 46
    pixels = np.asarray(rgb).astype(np.float32)
    blended = pixels * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def draw_boxes(base: Image.Image, boxes: list[dict]) -> Image.Image:
    image = base.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("segoeui.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for box in boxes[:15]:
        x1, y1, x2, y2 = box["xyxy"]
        label = f"{box['label']} {box['confidence']:.0%}"
        draw.rectangle((x1, y1, x2, y2), outline=(214, 122, 46), width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        tag_top = max(0, y1 - text_height - 8)
        draw.rectangle(
            (x1, tag_top, x1 + text_width + 10, tag_top + text_height + 6),
            fill=(214, 122, 46),
        )
        draw.text((x1 + 5, tag_top + 2), label, fill=(20, 16, 12), font=font)
    return image
