"""Look at the embedding results on this PC: summary, similar images, text search.

Writes <out-dir>/preview.html and opens it in the browser. The page shows images straight from
the local images folder (file:// links): nothing is uploaded anywhere.

    python client/inspect_results.py
    python client/inspect_results.py --server http://100.x.y.z:8080 --query "Logo auf blauem Grund"

--query needs the server (the search text is embedded with the same model, as a query) and
EMBED_API_KEY; without --query the script works offline.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import random
import sys
import webbrowser
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load(out_dir: Path) -> tuple[list[str], list[int], np.ndarray, set[str], int]:
    parts = sorted(out_dir.glob("part_*.parquet"))
    if not parts:
        sys.exit(f"No part_*.parquet in {out_dir}. Run client/embed_images.py first.")
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    shas = table.column("sha256").to_pylist()
    pages = table.column("page").to_pylist()
    dim = table.column("dim")[0].as_py()
    flat = table.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    vectors = np.asarray(flat, dtype=np.float32).reshape(-1, dim)
    models = set(table.column("model").to_pylist())
    return shas, pages, vectors, models, len(parts)


def image_path(images_dir: Path, sha: str, page: int) -> Path:
    name = "original.png" if page == 0 else f"p{page:03d}.png"
    return images_dir / sha / name


def figure(images_dir: Path, sha: str, page: int, caption: str) -> str:
    path = image_path(images_dir, sha, page)
    label = f"{sha[:10]}… {'original' if page == 0 else f'Seite {page}'}"
    return (
        f'<figure><a href="{path.as_uri()}" target="_blank">'
        f'<img src="{path.as_uri()}" loading="lazy"></a>'
        f"<figcaption>{html.escape(caption)}<br><small>{html.escape(label)}</small>"
        "</figcaption></figure>"
    )


def embed_queries(server: str, queries: list[str]) -> np.ndarray:
    key = os.environ.get("EMBED_API_KEY") or sys.exit("--query needs EMBED_API_KEY")
    r = httpx.post(
        f"{server.rstrip('/')}/embed",
        headers={"Authorization": f"Bearer {key}"},
        json={"texts": queries, "is_query": True},
        timeout=120,
    )
    if r.status_code != 200:
        sys.exit(f"Server error for --query: HTTP {r.status_code} {r.text[:200]}")
    return np.asarray(r.json()["embeddings"], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(r"C:\AI_Seminar_Images\embeddings"))
    p.add_argument("--images-dir", type=Path, default=Path(r"C:\AI_Seminar_Images\images"))
    p.add_argument("--samples", type=int, default=8, help="images to show neighbours for")
    p.add_argument("--neighbors", type=int, default=5, help="neighbours per image / hits per query")
    p.add_argument("--server", help="needed for --query, e.g. http://100.x.y.z:8080")
    p.add_argument("--query", action="append", default=[], help="search text (repeatable)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = p.parse_args()

    shas, pages, vectors, models, n_parts = load(args.out)
    norms = np.linalg.norm(vectors, axis=1)
    keys = set(zip(shas, pages, strict=True))
    errors = 0
    if (args.out / "errors.csv").exists():
        with (args.out / "errors.csv").open(encoding="utf-8") as fh:
            errors = sum(1 for _ in csv.DictReader(fh))

    print(f"Part files:        {n_parts}")
    print(f"Vectors:           {len(vectors)}  (unique images: {len(keys)})")
    print(f"Model:             {', '.join(sorted(models))}")
    print(f"Dimensions:        {vectors.shape[1]}")
    print(f"Norm min / max:    {norms.min():.4f} / {norms.max():.4f}  (expected 1.0000)")
    print(f"Image files:       {sum(1 for pg in pages if pg == 0)}")
    print(f"Document pages:    {sum(1 for pg in pages if pg > 0)}")
    print(f"Unreadable images: {errors}  (errors.csv)")
    if len(keys) != len(vectors):
        print("WARNING: some images appear twice in the part files")

    sections = []
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(vectors)), min(args.samples, len(vectors)))
    for i in picks:
        sims = vectors @ vectors[i]
        order = [j for j in np.argsort(-sims) if shas[j] != shas[i]][: args.neighbors]
        row = [figure(args.images_dir, shas[i], pages[i], "Ausgangsbild")]
        row += [
            figure(args.images_dir, shas[j], pages[j], f"Ähnlichkeit {sims[j]:.3f}") for j in order
        ]
        sections.append(f'<div class="row">{"".join(row)}</div>')

    query_sections = []
    if args.query:
        if not args.server:
            sys.exit("--query needs --server")
        for text, qvec in zip(args.query, embed_queries(args.server, args.query), strict=True):
            sims = vectors @ qvec
            hits = np.argsort(-sims)[: args.neighbors]
            row = [
                figure(args.images_dir, shas[j], pages[j], f"Treffer {sims[j]:.3f}") for j in hits
            ]
            query_sections.append(
                f"<h3>Suche: „{html.escape(text)}“</h3><div class='row'>{''.join(row)}</div>"
            )
            print(f"Query '{text}': best score {sims[hits[0]]:.3f}")

    page = f"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
<title>Embedding-Vorschau</title><style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
.row {{ display: flex; gap: 12px; margin: 8px 0 28px; overflow-x: auto; }}
figure {{ margin: 0; width: 220px; flex: none; background: #fff; padding: 8px;
  border: 1px solid #ddd; border-radius: 6px; }}
figure:first-child {{ border: 2px solid #c33; }}
img {{ width: 100%; height: 160px; object-fit: contain; background: #eee; }}
figcaption {{ font-size: 13px; margin-top: 6px; }}
small {{ color: #777; }}
</style></head><body>
<h1>Embedding-Vorschau</h1>
<p>{len(vectors)} Vektoren · Modell {html.escape(", ".join(sorted(models)))} ·
{vectors.shape[1]} Dimensionen. Bilder werden direkt aus {html.escape(str(args.images_dir))}
angezeigt; diese Seite bleibt auf diesem PC.</p>
{"<h2>Textsuche</h2>" + "".join(query_sections) if query_sections else ""}
<h2>Ähnlichste Bilder</h2>
<p>Links rot umrandet das Ausgangsbild, rechts die ähnlichsten Bilder aus anderen Dateien
(Seiten desselben Dokuments sind ausgeblendet). Wenn das Modell gut arbeitet, sehen die
Treffer dem Ausgangsbild inhaltlich oder gestalterisch ähnlich.</p>
{"".join(sections)}
</body></html>"""
    preview = args.out / "preview.html"
    preview.write_text(page, encoding="utf-8")
    print(f"\nPreview: {preview}")
    if not args.no_open:
        webbrowser.open(preview.as_uri())


if __name__ == "__main__":
    main()
