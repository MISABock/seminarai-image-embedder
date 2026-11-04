"""HTTP service on Computer C: POST /embed turns images (base64) or texts into vectors.

Same request/response shape as the SeminarAI project's embedder service
(SeminarAI_Group02/embedder/embedder/server.py), so the orchestrator could use it later:

    POST /embed  {"texts": [...], "images_b64": [...], "is_query": false}
              -> {"embeddings": [[...], ...], "model": "...", "dim": 1024}

Texts come first in the result, then images, each in request order.
Images are decoded in memory only: nothing is written to disk and nothing about the content
is logged (counts and timings only).

Start (see README.md):  uvicorn server.app:app --env-file .env --host <tailscale-ip> --port 8080
"""

from __future__ import annotations

import base64
import binascii
import hmac
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel

from . import setup_state
from .embedder import (
    DEFAULT_INSTRUCTION,
    MAX_PIXELS,
    MIN_PIXELS,
    EmbedderConfig,
    FakeEmbedder,
    QwenVLEmbedder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("embedder")

MAX_ITEMS_PER_REQUEST = 64
MAX_IMAGE_BYTES = 30 * 1024 * 1024
# Pipeline images are at most 2048 px per side; refuse anything absurd (decompression bombs).
Image.MAX_IMAGE_PIXELS = 60_000_000

QUERY_INSTRUCTION = (
    "Retrieve images, slides or layouts from a brand agency's client archive "
    "that match the description."
)


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise RuntimeError(f"{name} is not set (copy .env.example to .env and fill it in)")
    return value


def load_config() -> EmbedderConfig:
    return EmbedderConfig(
        model_id=_env("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-8B"),
        revision=os.environ.get("EMBED_MODEL_REVISION") or None,
        quant=_env("EMBED_QUANT", "int8"),
        dim=int(_env("EMBED_DIM", "1024")),
        max_pixels=int(_env("EMBED_MAX_PIXELS", str(MAX_PIXELS))),
        min_pixels=int(_env("EMBED_MIN_PIXELS", str(MIN_PIXELS))),
        batch_size=int(_env("EMBED_BATCH_SIZE", "4")),
        doc_instruction=_env("EMBED_DOC_INSTRUCTION", DEFAULT_INSTRUCTION),
        query_instruction=_env("EMBED_QUERY_INSTRUCTION", QUERY_INSTRUCTION),
    )


class State:
    embedder: QwenVLEmbedder | FakeEmbedder | None = None
    cfg: EmbedderConfig | None = None
    fake: bool = False
    api_key: str = ""
    lock = threading.Lock()  # one GPU job at a time


state = State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.api_key = _env("EMBED_API_KEY")
    if len(state.api_key) < 16:
        raise RuntimeError("EMBED_API_KEY is too short (use at least 16 random characters)")
    state.cfg = load_config()
    state.fake = os.environ.get("EMBED_FAKE") == "1"
    if state.fake:
        log.warning("EMBED_FAKE=1: random vectors, no model. Only for testing the connection.")
        state.embedder = FakeEmbedder(state.cfg)
    else:
        setup_state.check(state.cfg)  # step 2 only after a passed step 1 with these settings
        state.embedder = QwenVLEmbedder(state.cfg)
        log.info("READY: Schritt 2 laeuft, Bilder koennen jetzt vom PC gesendet werden.")
    yield


app = FastAPI(title="SeminarAI image embedder", lifespan=lifespan)


def require_key(authorization: str = Header(default="")) -> None:
    expected = f"Bearer {state.api_key}"
    if not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid API key")


class EmbedRequest(BaseModel):
    texts: list[str] | None = None
    images_b64: list[str] | None = None
    is_query: bool = False


def _model_tag() -> str:
    assert state.cfg is not None
    return "FAKE" if state.fake else state.cfg.model_tag


@app.get("/health")
def health() -> dict:
    cfg = state.cfg
    info = {
        "status": "ok",
        "model": _model_tag(),
        "dim": cfg.dim if cfg else None,
        "quant": cfg.quant if cfg else None,
        "max_pixels": cfg.max_pixels if cfg else None,
        "commit": getattr(state.embedder, "commit", None),
        "fake": state.fake,
    }
    if not state.fake:
        import torch

        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_used_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    return info


def _decode(b64: str, index: int) -> Image.Image:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, f"images_b64[{index}] is not valid base64") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"images_b64[{index}] is larger than {MAX_IMAGE_BYTES} bytes")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(400, f"images_b64[{index}] is not a readable image") from None
    return img.convert("RGB")


@app.post("/embed", dependencies=[Depends(require_key)])
def embed(req: EmbedRequest) -> dict:
    texts = req.texts or []
    images_b64 = req.images_b64 or []
    if not (texts or images_b64):
        raise HTTPException(400, "texts or images_b64 required")
    if len(texts) + len(images_b64) > MAX_ITEMS_PER_REQUEST:
        raise HTTPException(413, f"at most {MAX_ITEMS_PER_REQUEST} items per request")

    images = [_decode(b, i) for i, b in enumerate(images_b64)]
    started = time.monotonic()
    with state.lock:
        assert state.embedder is not None
        vectors = []
        if texts:
            vectors += state.embedder.embed_texts(texts, is_query=req.is_query)
        if images:
            vectors += state.embedder.embed_images(images)
    log.info(
        "embedded %d texts, %d images in %.2f s",
        len(texts),
        len(images),
        time.monotonic() - started,
    )
    assert state.cfg is not None
    return {"embeddings": vectors, "model": _model_tag(), "dim": state.cfg.dim}
