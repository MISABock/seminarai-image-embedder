"""Qwen3-VL-Embedding on one GPU: images and texts -> L2-normalised vectors.

Follows the official reference implementation (github.com/QwenLM/Qwen3-VL-Embedding,
src/models/qwen3_vl_embedding.py):
- chat format: system message = instruction, user message = the image or the text
- image resizing by qwen-vl-utils (patch size 16, min/max pixels), processor with do_resize=False
- embedding = last hidden state of the last non-padding token (right padding), L2-normalised

Added here: optional 8-bit / 4-bit loading (bitsandbytes) so the 8B model fits into 16 GB of
VRAM, and Matryoshka truncation to `dim` before normalising.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import dataclass

from PIL import Image

log = logging.getLogger("embedder")

# Defaults of the reference implementation.
MAX_LENGTH = 8192
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 1800 * 32 * 32
IMAGE_PATCH_SIZE = 16
DEFAULT_INSTRUCTION = "Represent the user's input."


@dataclass(frozen=True)
class EmbedderConfig:
    model_id: str
    revision: str | None
    quant: str  # "int8" | "nf4" | "none"
    dim: int
    max_pixels: int
    min_pixels: int
    batch_size: int
    doc_instruction: str
    query_instruction: str

    @property
    def model_tag(self) -> str:
        """Model name stored with every vector. Includes the quantisation, because vectors of
        the same model loaded differently must not be mixed either."""
        return f"{self.model_id}@{self.quant}"


def _with_period(instruction: str) -> str:
    """Like the reference: strip, and end with punctuation."""
    text = instruction.strip()
    if text and not unicodedata.category(text[-1]).startswith("P"):
        text += "."
    return text


class QwenVLEmbedder:
    """Loads the model once; `embed_images` / `embed_texts` are thread-unsafe, call them from
    one worker at a time (the FastAPI app serialises calls with a lock)."""

    def __init__(self, cfg: EmbedderConfig):
        import torch
        from transformers import BitsAndBytesConfig, Qwen3VLProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel,
            Qwen3VLPreTrainedModel,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Check the NVIDIA driver and that PyTorch was installed "
                "from the cu128 index (RTX 50 cards need CUDA 12.8 or newer)."
            )

        class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
            """Same module layout as the reference (self.model = Qwen3VLModel), so the
            checkpoint's weight names load without any mapping."""

            def __init__(self, config):
                super().__init__(config)
                self.model = Qwen3VLModel(config)
                self.post_init()

        self.cfg = cfg
        self.torch = torch
        self.device = torch.device("cuda:0")

        quant_config = None
        if cfg.quant == "int8":
            # Keep the vision tower in bf16: it is small (~0.6B) and int8 hurts it most.
            quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual"])
        elif cfg.quant == "nf4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["visual"],
            )
        elif cfg.quant != "none":
            raise ValueError(f"EMBED_QUANT must be int8, nf4 or none, not {cfg.quant!r}")

        started = time.monotonic()
        log.info(
            "loading %s (revision %s, %s) ...", cfg.model_id, cfg.revision or "main", cfg.quant
        )
        self.model, loading = Qwen3VLForEmbedding.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            device_map={"": 0},
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        # Missing weights would be initialised randomly and every vector would be garbage.
        if loading.get("missing_keys"):
            missing = loading["missing_keys"]
            raise RuntimeError(
                f"{len(missing)} weights missing in the checkpoint (e.g. {missing[:3]}): the "
                "module layout does not match this transformers version. Do not embed anything."
            )
        if loading.get("unexpected_keys"):
            log.warning("ignored %d unexpected weights", len(loading["unexpected_keys"]))
        self.model.eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            cfg.model_id, revision=cfg.revision, padding_side="right"
        )
        self.processor.tokenizer.padding_side = "right"  # pooling below relies on it

        hidden = self.model.config.text_config.hidden_size
        if not 64 <= cfg.dim <= hidden:
            raise ValueError(f"EMBED_DIM must be between 64 and {hidden}, not {cfg.dim}")
        self.commit = getattr(self.model.config, "_commit_hash", None)
        log.info(
            "model ready in %.0f s, %.1f GB VRAM in use, commit %s",
            time.monotonic() - started,
            torch.cuda.memory_allocated() / 1e9,
            self.commit,
        )

    # ------------------------------------------------------------------ public

    def embed_images(self, images: list[Image.Image], instruction: str | None = None):
        instr = _with_period(instruction or self.cfg.doc_instruction)
        items = [
            {
                "type": "image",
                "image": img,
                "min_pixels": self.cfg.min_pixels,
                "max_pixels": self.cfg.max_pixels,
            }
            for img in images
        ]
        return self._embed_items(items, instr)

    def embed_texts(self, texts: list[str], is_query: bool = False):
        instr = _with_period(self.cfg.query_instruction if is_query else self.cfg.doc_instruction)
        items = [{"type": "text", "text": t} for t in texts]
        return self._embed_items(items, instr)

    # ------------------------------------------------------------------ internals

    def _embed_items(self, items: list[dict], instruction: str):
        out = []
        for start in range(0, len(items), self.cfg.batch_size):
            out.extend(self._embed_batch(items[start : start + self.cfg.batch_size], instruction))
        return out

    def _embed_batch(self, items: list[dict], instruction: str) -> list[list[float]]:
        from qwen_vl_utils import process_vision_info

        torch = self.torch
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": [item]},
            ]
            for item in items
        ]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        images, _videos = process_vision_info(conversations, image_patch_size=IMAGE_PATCH_SIZE)
        inputs = self.processor(
            text=text,
            images=images,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            do_resize=False,  # qwen-vl-utils already resized to multiples of 32 px
            return_tensors="pt",
        ).to(self.device)

        forward_args = {
            k: inputs[k]
            for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
            if k in inputs
        }
        with torch.inference_mode():
            hidden = self.model.model(**forward_args).last_hidden_state
            mask = inputs["attention_mask"]
            last = mask.sum(dim=1) - 1  # right padding: index of the last real token
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            pooled = pooled.float()[:, : self.cfg.dim]  # Matryoshka: keep the first dims
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled.cpu().tolist()


class FakeEmbedder:
    """No GPU, no model: deterministic random unit vectors. For testing the network path
    (Tailscale, firewall, API key) before downloading 17 GB."""

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.commit = "fake"

    def _vector(self, payload: bytes) -> list[float]:
        import hashlib
        import random

        rng = random.Random(hashlib.sha256(payload).digest())
        v = [rng.gauss(0.0, 1.0) for _ in range(self.cfg.dim)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v]

    def embed_images(self, images: list[Image.Image], instruction: str | None = None):
        return [self._vector(img.tobytes()[:65536]) for img in images]

    def embed_texts(self, texts: list[str], is_query: bool = False):
        return [self._vector(t.encode()) for t in texts]
