# Kicau — Offline Bird-Call Identifier

**Kenali kicauan burung tanpa internet.** Kicau records or uploads bird audio and
identifies the species on-device, using the open BirdNET v2.4 weights. No
server, no API key, no account — it works in the middle of a national park with
zero signal, and your recordings never leave your device.

Built for the **Hacktoberfest 2026 Open-Source AI Challenge (Week 1: Touch
Grass)** — software that gets people off the screen and into the world.

## Why open AI is the whole point

- **Runs with no internet.** The model is a 26 MB TFLite file executed locally
  via LiteRT. Closed APIs can't identify a bird where there's no signal.
- **Data stays yours.** Recordings are processed on-device and deleted after
  analysis. Nothing is uploaded anywhere, ever.
- **Costs nothing to run.** No per-request fees, no quota. A student birding
  club can run it on a laptop forever.
- **Community labels.** Species names ship in English and Indonesian (from the BirdNET
  team provides them), so a kid in Java reads "Cikrak bambu", not just
  *Abroscopus superciliaris*.

## Quick start

Requirements: Python 3.10+, `ffmpeg` on PATH.

```bash
git clone <this repo> && cd kicau
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py   # one-time, ~53 MB, then you're offline forever

# CLI
python -m kicau.identify recording.mp3 --top-n 3

# Web UI (record from mic or upload a file)
python -m kicau.server
# open http://127.0.0.1:8765
```

## How it works

```
audio file ──▶ ffmpeg (decode → 48 kHz mono)
           ──▶ 3-second chunks (zero-padded tail)
           ──▶ BirdNET v2.4 TFLite fp16, via LiteRT (on-device)
           ──▶ flat sigmoid (same activation BirdNET-Analyzer uses)
           ──▶ top-N species per chunk, English/Indonesian common names ──▶ × location likelihood (optional)
```

The preprocessing and activation math mirror BirdNET-Analyzer v2.4 exactly
(chunk length, sample rate, sigmoid sensitivity), so results match the
reference implementation.

## Project layout

| Path | What |
|---|---|
| `kicau/identify.py` | Model loading, chunking, inference, EN/ID labels, location filter |
| `kicau/location.py` | BirdNET meta-model: (lat, lon, week) → per-species occurrence |
| `kicau/audio.py` | ffmpeg-based decode/resample |
| `kicau/server.py` | stdlib-only HTTP server + `/api/identify` |
| `kicau/labels_id.txt` | 6,522 species labels (BirdNET team; 1,227 with Indonesian names, rest fall back to English) |
| `kicau/labels_en.txt` | 6,522 species labels (English, BirdNET team) |
| `web/index.html` | Single-file offline UI, no CDN |
| `scripts/download_model.py` | One-time weight download |
| `tests/` | Pipeline tests (synthetic + real-audio path) |

## Model & data credits

- **BirdNET v2.4** — Kahl et al., Cornell Lab of Ornithology & Chemnitz
  University of Technology. Model weights: **CC-BY-NC-4.0** (free for
  research, education, and personal use). Analyzer code: MIT.
- Species labels: BirdNET team (`BirdNET-Analyzer`, MIT).
- This repo's code: MIT (see `LICENSE`).

## Contributing

See `CONTRIBUTING.md`. Ideas welcome: more Indonesian label coverage, a
PWA service worker for true offline-first install, on-device recording
directly in the CLI.
