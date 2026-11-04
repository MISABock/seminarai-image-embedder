"""Send the images from this PC to the embedder on Computer C and store the vectors in parts.

Input:  the `images` folder copied from the VM (processed_data/images), laid out as
        <images-dir>/<sha256>/original.png   -> page 0  (image files)
        <images-dir>/<sha256>/p001.png ...   -> page 1… (page / slide renders)
Output: <out-dir>/part_0001.parquet, part_0002.parquet, …
        columns: sha256, page, embedding (float32[dim]), model, dim
        <out-dir>/errors.csv  images the server could not read (sha256, page, reason)

Resumable: images already in a part file are skipped, so after a crash or Ctrl+C just start
the same command again. All parts must come from one model (same name and quantisation).

Example (PowerShell):
    $env:EMBED_API_KEY = "<key from Computer C>"
    python client/embed_images.py --server http://100.x.y.z:8080 --limit 100
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RENDER_RE = re.compile(r"^p(\d{3,})\.png$")
PART_RE = re.compile(r"^part_(\d{4,})\.parquet$")


@dataclass(frozen=True)
class Item:
    sha256: str
    page: int
    path: Path


def scan(images_dir: Path) -> list[Item]:
    """All images in the processed_data layout, sorted by hash and page."""
    items = []
    for folder in images_dir.iterdir():
        if not (folder.is_dir() and SHA_RE.match(folder.name)):
            continue
        for f in folder.iterdir():
            if f.name == "original.png":
                items.append(Item(folder.name, 0, f))
            elif m := RENDER_RE.match(f.name):
                items.append(Item(folder.name, int(m.group(1)), f))
    items.sort(key=lambda it: (it.sha256, it.page))
    return items


def existing_parts(out_dir: Path) -> tuple[set[tuple[str, int]], set[str], int]:
    """(done keys, model names found, highest part number)."""
    done: set[tuple[str, int]] = set()
    models: set[str] = set()
    highest = 0
    for f in sorted(out_dir.glob("part_*.parquet")):
        m = PART_RE.match(f.name)
        if not m:
            continue
        highest = max(highest, int(m.group(1)))
        table = pq.read_table(f, columns=["sha256", "page", "model"])
        models.update(table.column("model").unique().to_pylist())
        done.update(
            zip(table.column("sha256").to_pylist(), table.column("page").to_pylist(), strict=True)
        )
    return done, models, highest


def write_part(
    out_dir: Path, number: int, rows: list[tuple[Item, list[float]]], model: str, dim: int
):
    schema = pa.schema(
        [
            ("sha256", pa.string()),
            ("page", pa.int32()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("model", pa.string()),
            ("dim", pa.int32()),
        ]
    )
    table = pa.table(
        {
            "sha256": [it.sha256 for it, _ in rows],
            "page": [it.page for it, _ in rows],
            "embedding": [vec for _, vec in rows],
            "model": [model] * len(rows),
            "dim": [dim] * len(rows),
        },
        schema=schema,
    )
    final = out_dir / f"part_{number:04d}.parquet"
    tmp = final.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, final)  # a part is either complete or absent
    return final


def known_errors(out_dir: Path) -> set[tuple[str, int]]:
    path = out_dir / "errors.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {(row["sha256"], int(row["page"])) for row in csv.DictReader(fh)}


def log_errors(out_dir: Path, errors: list[tuple[Item, str]]) -> None:
    if not errors:
        return
    path = out_dir / "errors.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["sha256", "page", "reason"])
        for it, reason in errors:
            w.writerow([it.sha256, it.page, reason])


class Client:
    def __init__(self, server: str, api_key: str, timeout: float):
        self.http = httpx.Client(
            base_url=server.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=10),
        )

    def health(self) -> dict:
        r = self.http.get("/health")
        r.raise_for_status()
        return r.json()

    def embed(self, items: list[Item], retries: int = 6) -> list[list[float]]:
        payload = {"images_b64": [base64.b64encode(it.path.read_bytes()).decode() for it in items]}
        delay = 5.0
        for attempt in range(1, retries + 1):
            try:
                r = self.http.post("/embed", json=payload)
            except httpx.TransportError as exc:
                problem = f"connection problem ({type(exc).__name__})"
            else:
                if r.status_code == 200:
                    return r.json()["embeddings"]
                if r.status_code == 401:
                    sys.exit("Server says: invalid API key. Check EMBED_API_KEY.")
                if r.status_code in (400, 413):
                    raise BadInput(r.json().get("detail", r.text))
                problem = f"HTTP {r.status_code}"
            if attempt == retries:
                sys.exit(f"Giving up after {retries} attempts: {problem}. Start again to resume.")
            print(f"  {problem}, retry {attempt}/{retries - 1} in {delay:.0f} s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
        raise AssertionError("unreachable")


class BadInput(Exception):
    """The server refused the batch content (e.g. an unreadable image)."""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images-dir", type=Path, default=Path(r"C:\AI_Seminar_Images\images"))
    p.add_argument("--out", type=Path, default=Path(r"C:\AI_Seminar_Images\embeddings"))
    p.add_argument("--server", required=True, help="e.g. http://100.x.y.z:8080 (Computer C)")
    p.add_argument("--batch-size", type=int, default=8, help="images per request (default 8)")
    p.add_argument("--part-size", type=int, default=1000, help="images per part file")
    p.add_argument("--limit", type=int, default=0, help="stop after N new images (0 = all)")
    p.add_argument("--timeout", type=float, default=300, help="seconds per request")
    p.add_argument("--dry-run", action="store_true", help="only count, send nothing")
    p.add_argument("--retry-errors", action="store_true", help="send images from errors.csv again")
    args = p.parse_args()

    api_key = os.environ.get("EMBED_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit('Set the key first, e.g. PowerShell: $env:EMBED_API_KEY = "..."')
    if not args.images_dir.is_dir():
        sys.exit(f"{args.images_dir} does not exist")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.images_dir} ...", flush=True)
    items = scan(args.images_dir)
    done, models, last_part = existing_parts(args.out)
    failed = set() if args.retry_errors else known_errors(args.out)
    skip = done | failed
    pending = [it for it in items if (it.sha256, it.page) not in skip]
    if args.limit:
        pending = pending[: args.limit]
    total_bytes = sum(it.path.stat().st_size for it in pending)
    print(
        f"{len(items)} images found, {len(done)} already embedded, "
        f"{len(failed)} skipped (errors.csv), {len(pending)} to send "
        f"({total_bytes / 1e9:.1f} GB)",
        flush=True,
    )
    if args.dry_run or not pending:
        return

    client = Client(args.server, api_key, args.timeout)
    info = client.health()
    model, dim = info["model"], int(info["dim"])
    print(f"Server: model {model}, dim {dim}, gpu {info.get('gpu', '-')}", flush=True)
    if models and models != {model}:
        sys.exit(
            f"The existing parts were made with {sorted(models)}, the server runs {model}. "
            "Vectors of different models must not be mixed: use another --out folder."
        )
    if info.get("fake"):
        print("WARNING: the server runs in FAKE mode (random vectors). Test only!", flush=True)

    buffer: list[tuple[Item, list[float]]] = []
    part_no = last_part
    sent = 0
    finished = False
    started = time.monotonic()

    def flush() -> None:
        nonlocal buffer, part_no
        if buffer:
            part_no += 1
            path = write_part(args.out, part_no, buffer, model, dim)
            print(f"  saved {path.name} ({len(buffer)} images)", flush=True)
            buffer = []

    def take(batch: list[Item], vectors: list[list[float]]) -> None:
        if len(vectors) != len(batch) or any(len(v) != dim for v in vectors):
            sys.exit("Server returned the wrong number or size of vectors; stopping.")
        buffer.extend(zip(batch, vectors, strict=True))

    try:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            try:
                take(batch, client.embed(batch))
            except BadInput:
                # Find the bad image(s): send one by one, log and skip the ones refused.
                errors = []
                for it in batch:
                    try:
                        take([it], client.embed([it]))
                    except BadInput as exc:
                        errors.append((it, str(exc)))
                log_errors(args.out, errors)
                print(f"  {len(errors)} unreadable image(s) logged in errors.csv", flush=True)
            sent += len(batch)
            if len(buffer) >= args.part_size:
                flush()
            elapsed = time.monotonic() - started
            rate = sent / elapsed if elapsed else 0.0
            eta = (len(pending) - sent) / rate / 60 if rate else 0.0
            print(
                f"{sent}/{len(pending)}  {rate:.2f} img/s  ~{eta:.0f} min left",
                flush=True,
            )
        finished = True
    except KeyboardInterrupt:
        print("\nStopped by user, saving what is done ...", flush=True)
    finally:
        flush()
    if finished:
        print(f"Done: {sent} images sent. Results in {args.out}", flush=True)
    else:
        print("Interrupted. Start the same command again to continue.", flush=True)


if __name__ == "__main__":
    main()
