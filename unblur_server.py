#!/usr/bin/env python3
"""Real-ESRGAN powered backend for the Unblur app."""

from __future__ import annotations

import io
import logging
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import types
import math

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image
from torchvision.transforms import functional as tv_functional

# ---------------------------------------------------------------------------
# Compatibility patches
# ---------------------------------------------------------------------------
if "torchvision.transforms.functional_tensor" not in sys.modules:
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = tv_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# ---------------------------------------------------------------------------
# Logging & Flask setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent
MODEL_DIR = APP_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    urls: tuple[str, ...]
    scale: int
    tile: int
    pretty: str
    description: str
    model_builder: Callable[[], torch.nn.Module]


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "detail": ModelConfig(
        key="detail",
        name="RealESRGAN_x2plus",
        urls=(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        ),
        scale=2,
        tile=400,
        pretty="Detail Enhancer",
        description="Balanced restoration for photos, portraits, and general scenes.",
        model_builder=lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        ),
    ),
    "text": ModelConfig(
        key="text",
        name="realesr-general-x4v3",
        urls=(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/realesr-general-x4v3.pth",
            "https://huggingface.co/spaces/akhaliq/Real-ESRGAN/resolve/main/weights/realesr-general-x4v3.pth?download=1",
        ),
        scale=4,
        tile=300,
        pretty="Text & Logos",
        description="Sharper edge reconstruction tuned for text, UI, and graphic elements.",
        model_builder=lambda: SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        ),
    ),
}

DEFAULT_MODEL_KEY = "detail"


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
PROGRESS_LOCK = threading.Lock()
PROGRESS: Dict[str, Dict[str, Any]] = {}
PROGRESS_TTL_SECONDS = 600  # 10 minutes


def _cleanup_progress(now: Optional[float] = None) -> None:
    now = now or time.time()
    expired: list[str] = []
    with PROGRESS_LOCK:
        for job_id, payload in PROGRESS.items():
            updated = payload.get("updated", 0.0)
            if now - updated > PROGRESS_TTL_SECONDS:
                expired.append(job_id)
        for job_id in expired:
            PROGRESS.pop(job_id, None)


def _initialize_progress(job_id: str, model_key: str) -> None:
    now = time.time()
    with PROGRESS_LOCK:
        PROGRESS[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "completed": 0,
            "total": 0,
            "model": model_key,
            "message": None,
            "updated": now,
        }


def _update_progress(job_id: str, **updates: Any) -> None:
    now = time.time()
    with PROGRESS_LOCK:
        payload = PROGRESS.get(job_id)
        if not payload:
            return
        payload.update(updates)
        payload["updated"] = now


def _progress_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    with PROGRESS_LOCK:
        payload = PROGRESS.get(job_id)
        if not payload:
            return None
        return dict(payload)


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
logger.info("Using device: %s", DEVICE)


def download_model(config: ModelConfig) -> Path:
    """Ensure a Real-ESRGAN model is available locally and return its path."""
    model_path = MODEL_DIR / f"{config.name}.pth"
    if model_path.exists():
        return model_path

    last_error: Optional[Exception] = None
    for url in config.urls:
        logger.info("Downloading %s weights from %s", config.pretty, url)
        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info("Successfully downloaded %s", config.pretty)
            return model_path
        except Exception as exc:  # pragma: no cover - network failure
            last_error = exc
            logger.warning("Failed to download from %s (%s)", url, exc)
            if model_path.exists():
                model_path.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to download weights for {config.pretty}.") from last_error

    return model_path


def build_upsampler(model_key: str, tile_override: Optional[int] = None) -> RealESRGANer:
    """Create a Real-ESRGAN upsampler for the requested model."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")

    config = MODEL_CONFIGS[model_key]
    model_path = download_model(config)
    tile = tile_override or config.tile
    half_precision = DEVICE.type == "cuda"

    model = config.model_builder()

    upsampler = RealESRGANer(
        scale=config.scale,
        model_path=str(model_path),
        model=model,
        tile=max(64, tile),
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
        device=str(DEVICE),
    )

    logger.info(
        "Model loaded: %s (scale=%s, tile=%s, half_precision=%s)",
        config.pretty,
        config.scale,
        tile,
        half_precision,
    )

    return upsampler


UPSAMPLERS: Dict[str, RealESRGANer] = {}


def get_upsampler(model_key: str) -> RealESRGANer:
    """Return a ready-to-use upsampler, creating it if required."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")

    if model_key not in UPSAMPLERS:
        UPSAMPLERS[model_key] = build_upsampler(model_key)
    return UPSAMPLERS[model_key]


def _tile_process_with_progress(upsampler: RealESRGANer, job_id: str) -> None:
    batch, channel, height, width = upsampler.img.shape
    output_height = height * upsampler.scale
    output_width = width * upsampler.scale
    upsampler.output = upsampler.img.new_zeros((batch, channel, output_height, output_width))

    tile_size = max(upsampler.tile_size, 1)
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    total_tiles = max(1, tiles_x * tiles_y)
    _update_progress(job_id, total=total_tiles, completed=0, progress=0.0)

    completed = 0
    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            input_start_x_pad = max(input_start_x - upsampler.tile_pad, 0)
            input_end_x_pad = min(input_end_x + upsampler.tile_pad, width)
            input_start_y_pad = max(input_start_y - upsampler.tile_pad, 0)
            input_end_y_pad = min(input_end_y + upsampler.tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            input_tile = upsampler.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            with torch.no_grad():
                output_tile = upsampler.model(input_tile)

            output_start_x = input_start_x * upsampler.scale
            output_end_x = input_end_x * upsampler.scale
            output_start_y = input_start_y * upsampler.scale
            output_end_y = input_end_y * upsampler.scale

            output_start_x_tile = (input_start_x - input_start_x_pad) * upsampler.scale
            output_end_x_tile = output_start_x_tile + input_tile_width * upsampler.scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * upsampler.scale
            output_end_y_tile = output_start_y_tile + input_tile_height * upsampler.scale

            upsampler.output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :,
                :,
                output_start_y_tile:output_end_y_tile,
                output_start_x_tile:output_end_x_tile,
            ]

            completed += 1
            progress_value = min(0.99, completed / total_tiles)
            _update_progress(job_id, completed=completed, progress=progress_value)


def _enhance_with_progress(
    upsampler: RealESRGANer,
    img: np.ndarray,
    model_key: str,
    job_id: str,
) -> np.ndarray:
    config = MODEL_CONFIGS[model_key]
    h_input, w_input = img.shape[0:2]
    img = img.astype(np.float32)
    max_range = 65535 if np.max(img) > 256 else 255
    img = img / max_range

    if len(img.shape) == 2:
        img_mode = "L"
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img_mode = "RGBA"
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_mode = "RGB"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    upsampler.pre_process(img)
    if upsampler.tile_size > 0:
        _tile_process_with_progress(upsampler, job_id)
    else:
        upsampler.process()
        _update_progress(job_id, total=1, completed=1, progress=0.95)
    output_img = upsampler.post_process()
    output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    if img_mode == "L":
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    if img_mode == "RGBA":
        upsampler.pre_process(alpha)
        if upsampler.tile_size > 0:
            upsampler.tile_process()
        else:
            upsampler.process()
        output_alpha = upsampler.post_process()
        output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
        output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    if max_range == 65535:
        output = (output_img * 65535.0).round().astype(np.uint16)
    else:
        output = (output_img * 255.0).round().astype(np.uint8)

    if config.scale != float(upsampler.scale):
        output = cv2.resize(
            output,
            (int(w_input * config.scale), int(h_input * config.scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )

    return output


def enhance_image(img: np.ndarray, model_key: str, job_id: str) -> np.ndarray:
    """Apply the selected Real-ESRGAN model to the provided RGB numpy array."""
    config = MODEL_CONFIGS[model_key]
    upsampler = get_upsampler(model_key)
    try:
        return _enhance_with_progress(upsampler, img, model_key, job_id)
    except RuntimeError as err:
        logger.warning("Real-ESRGAN tiling fallback (%s) due to: %s", config.pretty, err)
        smaller_tile = max(128, config.tile // 2)
        _update_progress(job_id, message="Retrying with smaller tiles", completed=0, total=0, progress=0.0)
        upsampler = build_upsampler(model_key, tile_override=smaller_tile)
        UPSAMPLERS[model_key] = upsampler
        return _enhance_with_progress(upsampler, img, model_key, job_id)


def unblur_image(image_path: str, model_key: str, job_id: str) -> np.ndarray:
    """Run the selected Real-ESRGAN model and return a restored RGB array."""
    logger.info("Processing image %s with model %s (job=%s)", image_path, model_key, job_id)

    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    img_np = np.array(img)

    enhanced = enhance_image(img_np, model_key, job_id)

    enhanced_img = Image.fromarray(enhanced).resize(
        original_size, Image.Resampling.LANCZOS
    )

    logger.info("Image processing complete for job %s", job_id)
    return np.array(enhanced_img)


@app.route("/api/health", methods=["GET"])
def healthcheck():
    """Simple health endpoint for readiness probes."""
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "models": {
                key: {
                    "name": config.pretty,
                    "description": config.description,
                    "scale": config.scale,
                }
                for key, config in MODEL_CONFIGS.items()
            },
        }
    )


@app.route("/api/unblur", methods=["POST"])
def unblur_endpoint():
    """Accept an uploaded image, run Real-ESRGAN, and return the result."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    model_key = request.form.get("model", DEFAULT_MODEL_KEY)
    job_id = request.form.get("jobId") or str(uuid.uuid4())

    if model_key not in MODEL_CONFIGS:
        return jsonify({"error": f"Unknown model '{model_key}'"}), 400

    _cleanup_progress()
    _initialize_progress(job_id, model_key)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_path = tmp_file.name
        uploaded_file.save(tmp_path)

    try:
        result_rgb = unblur_image(tmp_path, model_key, job_id)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        success, encoded_img = cv2.imencode(
            ".jpg", result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if not success:
            raise RuntimeError("Failed to encode processed image")

        snapshot = _progress_snapshot(job_id) or {}
        completed_tiles = snapshot.get("total", snapshot.get("completed", 0))
        _update_progress(job_id, status="completed", progress=1.0, completed=completed_tiles)

        response = send_file(
            io.BytesIO(encoded_img.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
        )
        response.headers["X-Job-Id"] = job_id
        return response
    except Exception as exc:  # pragma: no cover - runtime failure
        logger.exception("Image processing failed: %s", exc)
        _update_progress(job_id, status="error", message=str(exc))
        return jsonify({"error": "Processing failed", "detail": str(exc)}), 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.route("/api/progress/<job_id>", methods=["GET"])
def progress_endpoint(job_id: str):
    """Return progress information for a running job."""
    _cleanup_progress()
    snapshot = _progress_snapshot(job_id)
    if snapshot is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": snapshot.get("status", "pending"),
        "progress": snapshot.get("progress", 0.0),
        "completed": snapshot.get("completed", 0),
        "total": snapshot.get("total", 0),
        "model": snapshot.get("model"),
        "message": snapshot.get("message"),
        "updated": snapshot.get("updated"),
    })


if __name__ == "__main__":
    logger.info("Starting Real-ESRGAN backend on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
