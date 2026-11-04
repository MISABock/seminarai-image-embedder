"""Schritt 1 auf Computer C: Einrichtung prüfen und Modell testen.

Prüft GPU, Pakete, .env, Tailscale und Firewall, lädt das Modell und testet es mit selbst
erzeugten Bildern. Verarbeitet KEINE echten Bilder und startet keinen Server. Am Ende steht
eine deutliche Meldung, ob Computer C bereit ist. Nur dann wird .setup-complete.json
geschrieben, ohne die der Server für Schritt 2 nicht startet.

    .venv\\Scripts\\python -m server.selftest
    .venv\\Scripts\\python -m server.selftest --fake   # ohne GPU/Modell, nur den Ablauf testen
"""

from __future__ import annotations

import argparse
import logging
import math
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
TOTAL_IMAGES = 52_362  # images in processed_data/images on the VM (2026-11-04)
FIREWALL_RULE = "SeminarAI Embedder (Tailscale)"
PORT = 8080


class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []
        self.problems: list[str] = []
        self.blocking = False  # a problem that makes the model test pointless
        self.data: dict = {}

    def ok(self, label: str, detail: str) -> None:
        self.lines.append((label, detail))
        print(f"  [OK]     {label}: {detail}", flush=True)

    def fail(self, label: str, detail: str, blocking: bool = False) -> None:
        self.blocking = self.blocking or blocking
        self.lines.append((label, "FEHLT/FEHLER - " + detail))
        self.problems.append(f"{label}: {detail}")
        print(f"  [FEHLER] {label}: {detail}", flush=True)


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "nicht installiert"


def _image(w: int, h: int, variant: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (240, 240, 235))
    d = ImageDraw.Draw(img)
    if variant == 0:  # red block and blue circle
        d.rectangle((w // 16, h // 9, w // 2, h * 8 // 9), fill=(200, 30, 45))
        d.ellipse((w * 3 // 5, h // 4, w * 9 // 10, h * 3 // 4), fill=(20, 60, 160))
    else:  # green stripes
        for x in range(0, w, w // 10):
            d.rectangle((x, 0, x + w // 20, h), fill=(30, 150, 60))
    return img


def check_environment(r: Report, fake: bool) -> None:
    import os

    print("\n1) Umgebung", flush=True)
    if (REPO / ".env").exists():
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env", override=False)
        r.ok(".env", "vorhanden")
    else:
        r.fail(".env", "fehlt: copy .env.example .env (README C5)", blocking=True)

    key = os.environ.get("EMBED_API_KEY", "")
    if len(key) >= 16 and key != "CHANGE_ME":
        r.ok("API-Schlüssel", f"gesetzt ({len(key)} Zeichen)")
    else:
        r.fail("API-Schlüssel", "EMBED_API_KEY in .env fehlt oder ist zu kurz (README C5)")
    if os.environ.get("EMBED_FAKE") == "1":
        r.fail("EMBED_FAKE", "steht auf 1: für den echten Betrieb in .env auf 0 setzen")

    versions = {p: _version(p) for p in ("torch", "transformers", "bitsandbytes", "qwen-vl-utils")}
    r.data["versions"] = versions
    missing = [p for p, v in versions.items() if v == "nicht installiert"]
    if missing and not fake:
        r.fail("Pakete", f"fehlen: {', '.join(missing)} (README C3/C4)", blocking=True)
    else:
        r.ok("Pakete", ", ".join(f"{p} {v}" for p, v in versions.items()))

    ip = _run(["tailscale", "ip", "-4"])
    if ip:
        r.data["tailscale_ip"] = ip.splitlines()[0]
        r.ok("Tailscale", f"IP {r.data['tailscale_ip']}")
    else:
        r.fail("Tailscale", "keine IP gefunden: läuft Tailscale und ist der Rechner angemeldet?")

    if platform.system() == "Windows":
        enabled = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-NetFirewallRule -DisplayName '{FIREWALL_RULE}' "
                "-ErrorAction SilentlyContinue).Enabled",
            ]
        )
        if enabled and "True" in enabled:
            r.ok("Firewall", f"Regel '{FIREWALL_RULE}' aktiv")
        else:
            r.fail("Firewall", "Regel fehlt: der Nutzer legt sie in einer Admin-PowerShell an (C6)")


def check_gpu(r: Report) -> None:
    print("\n2) Grafikkarte", flush=True)
    import torch

    if not torch.cuda.is_available():
        r.fail(
            "CUDA",
            "nicht verfügbar: PyTorch aus dem cu128-Index installieren (README C3)",
            blocking=True,
        )
        return
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9
    r.data["gpu"] = {"name": props.name, "vram_gb": round(total, 1)}
    r.ok("GPU", f"{props.name}, {total:.1f} GB, Compute {props.major}.{props.minor}")
    if total < 15:
        r.fail("VRAM", f"nur {total:.1f} GB: für 8B in 8 Bit werden ~16 GB erwartet")


def check_model(r: Report, fake: bool) -> None:
    print("\n3) Modell laden und testen (nur künstliche Testbilder)", flush=True)
    from .app import load_config
    from .embedder import FakeEmbedder, QwenVLEmbedder

    cfg = load_config()
    r.data["config"] = cfg
    started = time.monotonic()
    try:
        emb = FakeEmbedder(cfg) if fake else QwenVLEmbedder(cfg)
    except Exception as exc:  # show the cause and stop: without a model nothing else matters
        r.fail("Modell", f"{type(exc).__name__}: {exc}")
        return
    r.ok("Modell", f"{cfg.model_tag} geladen in {time.monotonic() - started:.0f} s")
    r.data["commit"] = getattr(emb, "commit", None)

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    img_a, img_b = _image(1280, 720, 0), _image(1280, 720, 1)
    va, vb = emb.embed_images([img_a, img_b])
    match, other = emb.embed_texts(
        [
            "a red rectangle and a blue circle on a light background",
            "a page of dense black text about tax law",
        ],
        is_query=True,
    )
    norm = math.sqrt(cos(va, va))
    if len(va) != cfg.dim or abs(norm - 1) > 1e-3:
        r.fail("Vektoren", f"Länge {len(va)} (erwartet {cfg.dim}), Norm {norm:.4f}")
        return
    r.ok("Vektoren", f"{len(va)} Zahlen, Norm {norm:.4f}")
    sim_match, sim_other, sim_ab = cos(va, match), cos(va, other), cos(va, vb)
    detail = f"passend {sim_match:.3f} / unpassend {sim_other:.3f} / anderes Bild {sim_ab:.3f}"
    if fake:
        r.ok("Ähnlichkeit", detail + " (Fake: nicht aussagekräftig)")
    elif sim_match > sim_other and sim_ab < 0.999:
        r.ok("Ähnlichkeit", detail)
    else:
        r.fail("Ähnlichkeit", detail + ": Modell liefert unplausible Vektoren")
        return

    # Speed: like the real data, half large image files, half page renders.
    batch = [_image(2048, 1536, i % 2) for i in range(4)] + [
        _image(1280, 905, i % 2) for i in range(4)
    ]
    emb.embed_images(batch[:1])  # warm-up
    t0 = time.monotonic()
    emb.embed_images(batch)
    per_image = (time.monotonic() - t0) / len(batch)
    hours = per_image * TOTAL_IMAGES / 3600
    r.data["seconds_per_image"] = round(per_image, 3)
    r.data["estimated_hours"] = round(hours, 1)
    r.ok("Tempo", f"ca. {per_image:.2f} s pro Bild -> ca. {hours:.0f} h für {TOTAL_IMAGES} Bilder")

    if not fake:
        import torch

        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        r.data["vram_peak_gb"] = round(peak, 1)
        if peak > total * 0.95:
            r.fail("VRAM", f"Spitze {peak:.1f} von {total:.1f} GB: EMBED_BATCH_SIZE verkleinern")
        else:
            r.ok("VRAM", f"Spitze {peak:.1f} von {total:.1f} GB")


def banner(r: Report, fake: bool) -> None:
    line = "=" * 70
    print("\n" + line)
    if r.problems:
        print("  EINRICHTUNG NICHT ABGESCHLOSSEN - bitte zuerst beheben:")
        for p in r.problems:
            print(f"   - {p}")
    elif fake:
        print("  TESTLAUF (FAKE) OHNE MODELL - Ablauf funktioniert.")
        print("  Es wurde KEINE Freigabe für Schritt 2 geschrieben.")
    else:
        cfg = r.data["config"]
        print("  EINRICHTUNG ABGESCHLOSSEN - Computer C ist bereit")
        print(line)
        gpu = r.data.get("gpu", {})
        print(f"  GPU:            {gpu.get('name')} ({gpu.get('vram_gb')} GB)")
        print(f"  Modell:         {cfg.model_tag}, {cfg.dim} Dimensionen")
        print(f"  VRAM (Spitze):  {r.data.get('vram_peak_gb')} GB")
        print(
            f"  Tempo (Test):   {r.data['seconds_per_image']} s/Bild, "
            f"ca. {r.data['estimated_hours']} h für alle"
        )
        print(f"  Adresse:        http://{r.data.get('tailscale_ip')}:{PORT}")
        print("  Es wurden KEINE echten Bilder verarbeitet, der Server läuft NICHT.")
        print()
        print('  Schritt 2 erst auf Ansage: Claude sagen "Wir embedden jetzt".')
    print(line + "\a", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Schritt 1: Einrichtung prüfen (keine echten Bilder)")
    p.add_argument("--fake", action="store_true", help="ohne GPU und Modell, nur den Ablauf testen")
    args = p.parse_args()
    # Keep umlauts readable when the output is piped (e.g. read by Claude Code).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    r = Report()
    print(f"Schritt 1 - Einrichtung prüfen ({datetime.now():%Y-%m-%d %H:%M})", flush=True)
    check_environment(r, args.fake)
    if not args.fake:
        check_gpu(r)
    if not r.blocking:
        check_model(r, args.fake)

    if not r.problems and not args.fake:
        from . import setup_state

        report = {k: v for k, v in r.data.items() if k != "config"}
        report["finished_at"] = datetime.now().isoformat(timespec="seconds")
        path = setup_state.write(r.data["config"], report)
        print(f"\n  Freigabe für Schritt 2 gespeichert: {path.name}", flush=True)
    banner(r, args.fake)
    sys.exit(1 if r.problems else 0)


if __name__ == "__main__":
    main()
