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
9. **Code-Änderungen zurück ins Repo:** Musst du auf Computer C etwas am Code anpassen (z. B.
   wegen einer Bibliotheksversion), committe es mit einer klaren Nachricht und pushe es, nachdem
   der Nutzer zugestimmt hat. So arbeiten PC und Computer C mit demselben Stand. Der Code wurde
   ohne GPU nur im Testmodus geprüft; das echte Laden des Modells läuft auf Computer C zum
   ersten Mal.

## Computer C: zwei Schritte, niemals vermischen

Finde zuerst heraus, ob du auf Computer C bist: `nvidia-smi` zeigt eine RTX 50xx mit ~16 GB.

### Schritt 1: Einrichten (gleich nach dem Klonen)

Ziel: alles installiert, eingestellt und getestet. **Keine echten Bilder, kein laufender
Server.** Startet dich der Nutzer nach dem Klonen („Richte … ein“ o. ä.), machst du nur das hier.

| # | Tun (README) | Prüfen |
|---|---|---|
| 1 | C1: `nvidia-smi`, `tailscale ip -4`, `git`, `uv` | GPU mit ~16 GB sichtbar; Tailscale-IP `100.x.y.z` notieren |
| 2 | C3: venv + `torch==2.8.0`/`torchvision==0.23.0` aus **cu128** | `torch.cuda.is_available()` ist `True`, Gerätename RTX 50xx |
| 3 | C4: `uv pip install -r requirements-server.txt` | `python -c "import transformers, bitsandbytes, qwen_vl_utils"` ohne Fehler |
| 4 | C5: `.env` aus `.env.example`, Schlüssel erzeugen, `EMBED_FAKE=0` | Werte gesetzt |
| 5 | C6: Firewall-Befehl **dem Nutzer geben** (Admin-PowerShell) | Nutzer bestätigt |
| 6 | C9: Energiesparmodus: **der Nutzer** stellt „Nie“ ein | Nutzer bestätigt |
| 7 | C7: `.venv\Scripts\python -m server.selftest` (lädt beim ersten Mal ~17 GB) | Kasten am Ende: `EINRICHTUNG ABGESCHLOSSEN` |

Der Selbsttest prüft Umgebung, GPU, Tailscale und Firewall, lädt das Modell, testet es mit selbst
erzeugten Bildern, misst das Tempo und schreibt bei Erfolg `.setup-complete.json`. Ohne diese
Datei startet der Server für Schritt 2 nicht.

**Wenn der Kasten `EINRICHTUNG ABGESCHLOSSEN` zeigt:**
1. Zeig dem Nutzer den Kasten (GPU, VRAM, Tempo, geschätzte Dauer, Adresse).
2. Zeig ihm einmal den API-Schlüssel aus `.env`; er braucht ihn auf dem PC.
3. Sag klar: „Schritt 1 ist fertig, Computer C ist bereit. Es wurden keine echten Bilder
   verarbeitet und der Server läuft nicht. Sag mir Bescheid, wenn wir embedden.“
4. **Dann aufhören.** Keinen Server starten, nichts weiter tun.

**Wenn der Kasten `EINRICHTUNG NICHT ABGESCHLOSSEN` zeigt:** die aufgelisteten Punkte beheben
(Fehlerbilder unten; Firewall und Energie macht der Nutzer) und den Selbsttest wiederholen.
Meldet er „weights missing“ oder unplausible Ähnlichkeit: **anhalten**, dem Nutzer die Meldung
zeigen. Wahrscheinliche Ursache: eine `transformers`-Version mit anderem Modul-Layout; dann
`transformers==4.57.*` probieren und den Selbsttest wiederholen.

### Schritt 2: Embedden (nur auf ausdrückliche Ansage)

Erst wenn der Nutzer ausdrücklich sagt, dass jetzt embeddet wird (z. B. „Wir embedden jetzt“).

1. Prüfen, dass `.setup-complete.json` existiert. Sonst zurück zu Schritt 1.
2. Server in einem **sichtbaren** Terminal starten (README C8, mit der Tailscale-IP), damit der
   Nutzer das Log sieht und ihn mit `Strg+C` beenden kann.
3. Der Server prüft beim Start, dass Schritt 1 mit genau diesen Einstellungen bestanden wurde.
   Verweigert er den Start: **nicht umgehen** (nicht `.setup-complete.json` anlegen oder
   bearbeiten), sondern dem Nutzer den Grund nennen und nach Rückfrage Schritt 1 wiederholen.
4. Warten, bis im Log `READY: Schritt 2 laeuft` steht.
5. Dem Nutzer die Befehle für den PC geben, mit eingesetzter IP (README P3/P4): erst
   `check_server.py`, dann `embed_images.py --limit 100`, dann alles.
6. Während des Laufs das Log im Blick behalten (z. B. `CUDA out of memory`). Wenn alles fertig ist
   und der Nutzer zustimmt: Server mit `Strg+C` beenden.

### Fehlerbilder

| Meldung | Ursache / Abhilfe |
|---|---|
| `CUDA is not available` | Falscher PyTorch-Build (nicht cu128) oder alter Treiber; Schritt 1.2 wiederholen |
| `no kernel image is available for execution on the device` | PyTorch kennt Blackwell nicht: cu128-Build installieren |
| `CUDA out of memory` | `EMBED_BATCH_SIZE` auf 2 oder 1 (ändert die Vektoren nicht); Server bzw. Selbsttest neu starten |
| Fehler aus `bitsandbytes` beim Laden | `uv pip install -U bitsandbytes` |
| `cannot import name 'Qwen3VL…' from transformers` | `transformers` zu alt: `uv pip install "transformers>=4.57,<5"` |
| Server: „Einrichtung noch nicht abgeschlossen“ / „Einstellungen geändert“ | Schritt 1 (Selbsttest) wiederholen; nichts von Hand umgehen |
| PC erreicht C nicht (`check_server`: not reachable) | Server läuft? Richtige Tailscale-IP? Firewall-Regel? Beide Geräte in `tailscale status`? |

## Ablauf auf dem PC

1. Bilder vorhanden? `C:\AI_Seminar_Images\images` enthält Ordner mit 64-stelligen Hex-Namen.
   Wenn nicht: README P1 (ssh + tar von der VM `misa@100.122.206.2`, braucht die Passphrase des
   Nutzers; der Nutzer tippt sie selbst).
2. README P2: venv und `requirements-client.txt`.
3. README P3: `check_server.py` gegen Computer C.
4. README P4: `--dry-run`, dann `--limit 100`, dann alles. Lange Läufe in einem eigenen
   Terminal starten, damit der Nutzer sie sieht und mit `Strg+C` anhalten kann.
5. Nach dem Lauf: `python client\inspect_results.py` (README P5) zeigt Anzahl, Modell, Normen und
   Fehler und öffnet eine Vorschau mit ähnlichen Bildern; mit `--server … --query "…"` auch eine
   Textsuche.

## Code-Konventionen

- Python 3.12, `ruff` (Zeilenlänge 100, `ruff.toml`): `uvx ruff check . && uvx ruff format --check .`
- Schnittstelle `POST /embed` wie im SeminarAI-Projekt (`embedder/embedder/server.py`):
  `{"texts", "images_b64", "is_query"}` → `{"embeddings", "model", "dim"}`. Nicht ohne Grund ändern;
  der Orchestrator des Projekts soll den Dienst später für Suchanfragen nutzen können.
- Ohne GPU testen: `EMBED_FAKE=1` startet den Server ohne Modell (Zufallsvektoren, braucht keine
  Einrichtung); `python -m server.selftest --fake` prüft den Ablauf des Selbsttests.
