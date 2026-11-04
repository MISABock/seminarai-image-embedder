# SeminarAI Image Embedder

Bild-Embeddings für das SeminarAI-Projekt (Gruppe 02) mit **Qwen3-VL-Embedding-8B** auf einer
eigenen Grafikkarte. Die Bilder kommen aus `processed_data/images` der SeminarAI-VM.

```
PC (mit den Bildern)                       Tailscale            Computer C (Windows, RTX 50xx, 16 GB)
C:\AI_Seminar_Images\images\<hash>\…  ── Bilder ─────────────►  server/  (FastAPI + Qwen3-VL-Embedding-8B, 8 Bit)
client/embed_images.py                                            Bilder nur im Arbeitsspeicher, nichts auf Disk
C:\AI_Seminar_Images\embeddings\      ◄── Vektoren (1024 Zahlen) ─┘
  part_0001.parquet, part_0002.parquet …
```

- **Computer C** lädt das Modell einmal auf die Grafikkarte und beantwortet Anfragen an `POST /embed`.
- **Der PC** schickt die Bilder in kleinen Paketen und speichert die Vektoren laufend in
  Parquet-Dateien. Bricht etwas ab, startest du denselben Befehl einfach noch einmal: Fertige
  Bilder werden übersprungen.

Auf Computer C gibt es **zwei getrennte Schritte**:
1. **Einrichten** (C1–C7): installieren, einstellen, Modell mit selbst erzeugten Testbildern
   prüfen. Endet mit der Meldung „EINRICHTUNG ABGESCHLOSSEN“. Keine echten Bilder.
2. **Embedden** (C8): erst auf Ansage. Der Server startet nur, wenn Schritt 1 bestanden wurde.

Für Claude Code auf beiden Rechnern gibt es eine Schritt-für-Schritt-Anleitung in `CLAUDE.md`.

---

## Teil C: Computer C einrichten (Windows, RTX 50xx, 16 GB VRAM)

**C1 – Voraussetzungen prüfen**

```powershell
nvidia-smi          # zeigt die Grafikkarte mit ~16 GB? Treiber aktuell?
tailscale ip -4     # Computer C ist im selben Tailnet wie der PC? Diese IP brauchst du später.
git --version
uv --version        # fehlt: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**C2 – Repo holen**

Das Repo ist privat: Beim ersten `git clone` öffnet Git ein Browserfenster für die Anmeldung
bei GitHub (Konto `MISABock` oder ein Konto, das Zugriff auf das Repo hat).

```powershell
cd C:\
git clone https://github.com/MISABock/seminarai-image-embedder.git
cd seminarai-image-embedder
```

Danach in diesem Ordner Claude Code starten und sagen: *„Richte nach CLAUDE.md den
Embedding-Dienst auf diesem Rechner ein.“* Claude macht dann nur **Schritt 1** (C3–C7) und meldet
sich mit „EINRICHTUNG ABGESCHLOSSEN“. Embeddet wird erst, wenn du sagst: *„Wir embedden jetzt.“*

**C3 – Python und PyTorch für RTX 50 (CUDA 12.8)**

RTX-50-Karten brauchen PyTorch mit CUDA 12.8 oder neuer (`cu128`). Ältere Builds (`cu124`)
erkennen die Karte nicht.

```powershell
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Die letzte Zeile muss `True` und den Namen der Grafikkarte ausgeben.

**C4 – Restliche Pakete**

```powershell
uv pip install -r requirements-server.txt
```

**C5 – Einstellungen (`.env`)**

```powershell
copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Den ausgegebenen Schlüssel in `.env` bei `EMBED_API_KEY=` eintragen. Den gleichen Schlüssel
brauchst du auf dem PC. `.env` kommt nie ins Git.

**C6 – Windows-Firewall öffnen (nur für Tailscale)**

In einer **Administrator**-PowerShell:

```powershell
New-NetFirewallRule -DisplayName "SeminarAI Embedder (Tailscale)" -Direction Inbound -Protocol TCP -LocalPort 8080 -RemoteAddress 100.64.0.0/10 -Action Allow
```

Damit erreichen nur Geräte aus dem Tailscale-Netz den Dienst.

**C7 – Schritt 1 abschließen: Selbsttest**

```powershell
.venv\Scripts\python -m server.selftest
```

Prüft Umgebung, Grafikkarte, Tailscale und Firewall, lädt das Modell (beim ersten Mal ca. 17 GB
Download nach `HF_HOME`) und testet es mit **selbst erzeugten** Bildern. Es werden keine echten
Bilder verarbeitet und kein Server gestartet. Am Ende steht ein Kasten:

```
======================================================================
  EINRICHTUNG ABGESCHLOSSEN - Computer C ist bereit
======================================================================
  GPU:            NVIDIA GeForce RTX 50xx (16.0 GB)
  Modell:         Qwen/Qwen3-VL-Embedding-8B@int8, 1024 Dimensionen
  VRAM (Spitze):  …
  Tempo (Test):   … s/Bild, ca. … h für alle
  Adresse:        http://100.x.y.z:8080
  Es wurden KEINE echten Bilder verarbeitet, der Server läuft NICHT.
```

Nur dann wird `.setup-complete.json` geschrieben. Ohne diese Datei, oder wenn sich danach eine
Einstellung ändert, die die Vektoren beeinflusst, startet der Server nicht.

**C8 – Schritt 2: Server starten (erst wenn embeddet wird)**

```powershell
.venv\Scripts\python -m uvicorn server.app:app --env-file .env --host <TAILSCALE-IP> --port 8080
```

Nach 1–2 Minuten steht im Log `READY: Schritt 2 laeuft`. Dann auf dem PC Teil P3/P4 ausführen.
Beenden mit `Strg+C`.

**C9 – Wach bleiben**

Während des Embeddens darf Computer C nicht in den Energiesparmodus gehen (Einstellungen →
System → Netzbetrieb und Energie → „Nie“).

---

## Teil P: PC mit den Bildern

**P1 – Bilder von der VM holen** (ca. 42 GB, mehrere Stunden; das Fenster offen lassen)

```bat
mkdir C:\AI_Seminar_Images
cd C:\AI_Seminar_Images
ssh misa@100.122.206.2 "cd /srv/data/processed_data && tar -cf - images" | tar -xf -
```

Danach liegen die Bilder in `C:\AI_Seminar_Images\images\<hash>\original.png` bzw. `p001.png …`.
Nicht in einen OneDrive-Ordner legen.

**P2 – Python-Umgebung**

```powershell
cd C:\SoftwareProjects\AISEMINAR\seminarai-image-embedder
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements-client.txt
```

**P3 – Verbindung prüfen**

```powershell
$env:EMBED_API_KEY = "<schlüssel von Computer C>"
python client\check_server.py --server http://<TAILSCALE-IP-VON-C>:8080
```

**P4 – Zählen, dann klein testen, dann alles**

```powershell
python client\embed_images.py --server http://<TAILSCALE-IP-VON-C>:8080 --dry-run
python client\embed_images.py --server http://<TAILSCALE-IP-VON-C>:8080 --limit 100
python client\embed_images.py --server http://<TAILSCALE-IP-VON-C>:8080
```

Der Test mit 100 Bildern zeigt `img/s`. Daraus ergibt sich die Gesamtdauer. Abbrechen mit
`Strg+C` ist jederzeit möglich, fertige Pakete bleiben gespeichert.

Ergebnis: `C:\AI_Seminar_Images\embeddings\part_*.parquet` mit den Spalten `sha256`, `page`
(0 = `original.png`, 1… = `p001.png` …), `embedding` (1024 × float32, L2-normiert), `model`, `dim`.
Bilder, die der Server nicht lesen konnte, stehen in `errors.csv`.

---

## Wichtig: nichts nachträglich ändern

Alle Vektoren müssen mit **genau denselben Einstellungen** entstehen, sonst sind sie nicht
vergleichbar: `EMBED_MODEL`, `EMBED_MODEL_REVISION`, `EMBED_QUANT`, `EMBED_DIM`,
`EMBED_MAX_PIXELS` und die beiden Anweisungstexte. Das Skript weigert sich, Ergebnisse
verschiedener Modelle im selben Ordner zu mischen. Muss sich etwas ändern: neuer `--out`-Ordner
und alles neu embedden. `EMBED_BATCH_SIZE` darf geändert werden (z. B. bei „CUDA out of memory“).

Der Modellname in den Ergebnissen enthält die Quantisierung, z. B.
`Qwen/Qwen3-VL-Embedding-8B@int8`. Genau dieser Name muss später als `IMAGE_EMBED_MODEL` auf der
VM stehen, und **Suchanfragen müssen mit demselben Dienst** embeddet werden (`is_query: true`).

## Noch offen

- **Rückweg zur VM:** Die Parquet-Dateien sind nach `sha256` + `page` sortiert. Für den Import in
  `kb.images` (`pipeline import-embeddings`) braucht es noch ein kleines Umwandlungsskript, das
  sie über `items.parquet` eines Exports den `file_id`s zuordnet.
- **Teamentscheidung:** Das SeminarAI-Projekt hat bisher Qwen3-VL-Embedding-**2B** festgelegt
  (ADR 0004/0007). Der Wechsel auf 8B braucht ein ADR und einen Plan, wo die Suchanfragen im
  Chat embeddet werden.

## Grundlage

Vorverarbeitung und Pooling folgen der offiziellen Referenzimplementierung
[QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)
(Systemanweisung, Bildgröße über `qwen-vl-utils`, letzter Token, L2-Normierung). Neu sind hier
das Laden in 8 Bit und das Kürzen auf 1024 Dimensionen (Matryoshka).
