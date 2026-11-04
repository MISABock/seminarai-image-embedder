"""Quick check of the embedder: health, one synthetic image, one text query.

Uses no client data: the test image is generated here. Works on Computer C itself
(--server http://127.0.0.1:8080 or the Tailscale IP) and from the PC.

    $env:EMBED_API_KEY = "..."
    python client/check_server.py --server http://100.x.y.z:8080
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import os
import sys
import time

import httpx
from PIL import Image, ImageDraw


def synthetic_png() -> bytes:
    img = Image.new("RGB", (1280, 720), (240, 240, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 80, 600, 640), fill=(200, 30, 45))  # a red block, like a key visual
    draw.ellipse((760, 160, 1160, 560), fill=(20, 60, 160))  # a blue circle
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    args = p.parse_args()
    key = os.environ.get("EMBED_API_KEY") or sys.exit("EMBED_API_KEY is not set")

    http = httpx.Client(
        base_url=args.server.rstrip("/"),
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(300, connect=10),
    )
    try:
        info = http.get("/health").json()
    except httpx.TransportError as exc:
        sys.exit(f"Server not reachable: {exc}. Is it running? Firewall? Tailscale IP correct?")
    print("health:", info)

    t0 = time.monotonic()
    r = http.post(
        "/embed",
        json={"images_b64": [base64.b64encode(synthetic_png()).decode()]},
    )
    if r.status_code != 200:
        sys.exit(f"/embed with an image failed: HTTP {r.status_code} {r.text[:300]}")
    image_vec = r.json()["embeddings"][0]
    t1 = time.monotonic()

    r = http.post(
        "/embed",
        json={
            "texts": [
                "a red rectangle and a blue circle on a light background",
                "a page of dense black text about tax law",
            ],
            "is_query": True,
        },
    )
    if r.status_code != 200:
        sys.exit(f"/embed with a text failed: HTTP {r.status_code} {r.text[:300]}")
    match_vec, other_vec = r.json()["embeddings"]
    t2 = time.monotonic()

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    norm = math.sqrt(cos(image_vec, image_vec))
    sim_match, sim_other = cos(image_vec, match_vec), cos(image_vec, other_vec)
    print(f"image: dim {len(image_vec)}, norm {norm:.4f}, {t1 - t0:.2f} s")
    print(f"2 text queries: {t2 - t1:.2f} s")
    print(f"similarity image <-> matching text:  {sim_match:.3f}")
    print(f"similarity image <-> unrelated text: {sim_other:.3f}")

    shape_ok = len(image_vec) == info["dim"] and abs(norm - 1) < 1e-3
    if info.get("fake"):
        print("OK (fake mode: shapes only)" if shape_ok else "UNEXPECTED SHAPE")
    elif shape_ok and sim_match > sim_other:
        print("OK: the matching text is closer to the image than the unrelated one")
    else:
        print("UNEXPECTED: check the model loading log before embedding real images")


if __name__ == "__main__":
    main()
