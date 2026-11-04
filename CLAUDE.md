# CLAUDE.md

Anleitung für Claude Code auf **Computer C** (GPU-Rechner) und auf dem **PC mit den Bildern**.
Die ausführlichen Befehle stehen in `README.md`; hier steht, was du in welcher Reihenfolge tust,
was du prüfst und wann du anhältst.

## Worum es geht

Rund 52.000 Bilder (ca. 42 GB) aus dem SeminarAI-Projekt sollen mit
**Qwen3-VL-Embedding-8B** in Vektoren mit 1024 Zahlen umgewandelt werden.
- **Computer C**: Windows, NVIDIA RTX 50xx mit 16 GB VRAM. Führt `server/` aus (FastAPI), lädt das
  Modell in **8 Bit** (bitsandbytes), weil es in voller Genauigkeit (~16–18 GB) nicht passt.
- **PC**: hat die Bilder in `C:\AI_Seminar_Images\images\<sha256>\{original.png | p001.png …}`,
  führt `client/embed_images.py` aus und speichert die Ergebnisse in
  `C:\AI_Seminar_Images\embeddings\part_*.parquet`.
- Verbindung über **Tailscale**. Computer C ist nur über seine Tailscale-IP erreichbar.

Die Bilder sind Kundendaten einer Agentur (NDA). Der Besitzer des Repos darf sie auf dem PC und
auf Computer C verarbeiten.

## Regeln

1. **Sprich Deutsch und erkläre einfach.** Der Nutzer ist im Terminal noch unsicher: ein Befehl
   pro Codeblock, kurz sagen, was er bewirkt und was danach zu sehen sein sollte.
2. **Keine Kundendaten ins Git.** Keine Bilder, keine Parquet-Dateien, keine `errors.csv`, keine
   `.env`. `.gitignore` deckt das ab; nichts mit `-f` hinzufügen.
3. **Bilder nie auf Computer C speichern** und nie Bildinhalte oder Dateipfade loggen. Der Server
   hält Bilder nur im Arbeitsspeicher. Bei Änderungen am Server muss das so bleiben.
4. **Einstellungen nach dem ersten echten Lauf nicht mehr ändern:** `EMBED_MODEL`,
   `EMBED_MODEL_REVISION`, `EMBED_QUANT`, `EMBED_DIM`, `EMBED_MAX_PIXELS`,
   `EMBED_DOC_INSTRUCTION`, `EMBED_QUERY_INSTRUCTION`. Sonst passen alte und neue Vektoren nicht
   zusammen. Wenn es nötig ist: dem Nutzer erklären, neuer `--out`-Ordner, alles neu.
   `EMBED_BATCH_SIZE` darf geändert werden.
5. **Systemeinstellungen macht der Nutzer selbst:** Firewall-Regel, Energieoptionen,
   Treiber-Installation. Gib ihm den genauen Befehl (README C6/C9) und warte auf seine
   Bestätigung.
6. **Der API-Schlüssel** steht nur in `.env` auf Computer C und als Umgebungsvariable
   `EMBED_API_KEY` auf dem PC. Zeig ihn dem Nutzer einmal, damit er ihn auf den PC übertragen
   kann. Er gehört in keine Datei im Repo, kein Issue und keinen Chat außerhalb dieser Sitzung.
7. **Nur dieses Modell laden:** `Qwen/Qwen3-VL-Embedding-8B` von Hugging Face. Keine anderen
   Modelle oder Skripte aus dem Internet ausführen.
8. **Halte an und frag nach**, wenn ein Prüfschritt unten nicht das erwartete Ergebnis zeigt.
   Nicht mit Umwegen weitermachen, die die Vektoren verändern könnten.

## Ablauf auf Computer C

Finde zuerst heraus, ob du auf Computer C bist: `nvidia-smi` zeigt eine RTX 50xx mit ~16 GB.

| Schritt | Tun (README) | Prüfen |
|---|---|---|
| 1 | C1: `nvidia-smi`, `tailscale ip -4`, `git`, `uv` | GPU mit ~16 GB sichtbar; Tailscale-IP `100.x.y.z` notieren |
| 2 | C3: venv + `torch==2.8.0`/`torchvision==0.23.0` aus **cu128** | `torch.cuda.is_available()` ist `True`, Gerätename RTX 50xx |
| 3 | C4: `uv pip install -r requirements-server.txt` | `python -c "import transformers, bitsandbytes, qwen_vl_utils"` ohne Fehler |
| 4 | C5: `.env` aus `.env.example`, Schlüssel erzeugen | `EMBED_API_KEY` gesetzt, `EMBED_FAKE=1` für den Test |
| 5 | C6: Firewall-Befehl **dem Nutzer geben** (Admin-PowerShell) | Nutzer bestätigt |
| 6 | C7: Server im Testmodus starten (im Hintergrund) | `GET http://<IP>:8080/health` zeigt `"fake": true` |
| 7 | Nutzer prüft vom PC aus mit `client\check_server.py` | Ausgabe `OK (fake mode …)` |
| 8 | C8: `EMBED_FAKE=0`, Server neu starten | Log: `model ready in … s, … GB VRAM in use`; **kein** Fehler „weights missing“ |
| 9 | `client\check_server.py` gegen den echten Server | `OK: the matching text is closer …` |
| 10 | Nutzer startet auf dem PC `--limit 100` | `img/s` notieren, Gesamtdauer hochrechnen, dem Nutzer sagen |

Wenn Schritt 8 mit „weights missing“ abbricht oder Schritt 9 `UNEXPECTED` meldet: **nichts
embedden.** Dem Nutzer die Fehlermeldung zeigen. Wahrscheinliche Ursache: eine andere
`transformers`-Version als 4.57.x mit anderem Modul-Layout. Dann `transformers==4.57.*` probieren.

### Fehlerbilder

| Meldung | Ursache / Abhilfe |
|---|---|
| `CUDA is not available` | Falscher PyTorch-Build (nicht cu128) oder alter Treiber; Schritt 2 wiederholen |
| `no kernel image is available for execution on the device` | PyTorch kennt Blackwell nicht: cu128-Build installieren |
| `CUDA out of memory` | `EMBED_BATCH_SIZE` auf 2 oder 1; Server neu starten |
| Fehler aus `bitsandbytes` beim Laden | `uv pip install -U bitsandbytes` |
| `cannot import name 'Qwen3VL…' from transformers` | `transformers` zu alt: `uv pip install "transformers>=4.57,<5"` |
| PC erreicht C nicht (`check_server`: not reachable) | Server läuft? Richtige Tailscale-IP? Firewall-Regel (Schritt 5)? Beide Geräte in `tailscale status`? |

## Ablauf auf dem PC

1. Bilder vorhanden? `C:\AI_Seminar_Images\images` enthält Ordner mit 64-stelligen Hex-Namen.
   Wenn nicht: README P1 (ssh + tar von der VM `misa@100.122.206.2`, braucht die Passphrase des
   Nutzers; der Nutzer tippt sie selbst).
2. README P2: venv und `requirements-client.txt`.
3. README P3: `check_server.py` gegen Computer C.
4. README P4: `--dry-run`, dann `--limit 100`, dann alles. Lange Läufe in einem eigenen
   Terminal starten, damit der Nutzer sie sieht und mit `Strg+C` anhalten kann.
5. Nach dem Lauf: Anzahl Zeilen in allen `part_*.parquet` und Einträge in `errors.csv` nennen.

## Code-Konventionen

- Python 3.12, `ruff` (Zeilenlänge 100, `ruff.toml`): `uvx ruff check . && uvx ruff format --check .`
- Schnittstelle `POST /embed` wie im SeminarAI-Projekt (`embedder/embedder/server.py`):
  `{"texts", "images_b64", "is_query"}` → `{"embeddings", "model", "dim"}`. Nicht ohne Grund ändern;
  der Orchestrator des Projekts soll den Dienst später für Suchanfragen nutzen können.
- Ohne GPU testen: `EMBED_FAKE=1` startet den Server ohne Modell (Zufallsvektoren).
