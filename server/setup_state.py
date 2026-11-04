"""Phase 1 (setup) leaves a marker file; phase 2 (the real server) refuses to start without it
or when the vector-relevant settings changed since the setup self-test."""

from __future__ import annotations

import json
from pathlib import Path

from .embedder import EmbedderConfig

SETUP_FILE = Path(__file__).resolve().parents[1] / ".setup-complete.json"

PHASE1_HINT = "Erst Schritt 1 ausführen: python -m server.selftest (siehe CLAUDE.md)."


def write(cfg: EmbedderConfig, report: dict) -> Path:
    data = {"settings": cfg.fingerprint(), "report": report}
    SETUP_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return SETUP_FILE


def check(cfg: EmbedderConfig) -> None:
    """Raise RuntimeError (German message for the user) unless phase 1 passed with exactly
    these settings."""
    if not SETUP_FILE.exists():
        raise RuntimeError(f"Die Einrichtung ist noch nicht abgeschlossen. {PHASE1_HINT}")
    saved = json.loads(SETUP_FILE.read_text(encoding="utf-8")).get("settings", {})
    now = cfg.fingerprint()
    changed = sorted(k for k in now if saved.get(k) != now[k])
    if changed:
        raise RuntimeError(
            f"Diese Einstellungen haben sich seit der Einrichtung geändert: {', '.join(changed)}. "
            f"Vektoren mit anderen Einstellungen passen nicht zusammen. {PHASE1_HINT}"
        )
