# Contributing to Kicau

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
python scripts/download_model.py
pytest
```

## Ground rules

- **Offline-first.** Any feature that requires the network after the one-time
  model download needs a very good reason. The trail has no signal.
- **Small dependencies.** `requirements.txt` is `numpy` + `ai-edge-litert`.
  Think twice before adding anything heavier.
- **Indonesian-first UX.** User-facing strings in the web UI are Bahasa
  Indonesia; code, docs, and identifiers stay in English.
- **Verify before claiming.** If you touch the inference path, run the full
  test suite *and* the reference soundscape check below.

## Verifying the inference path

```bash
# BirdNET team's own test soundscape — expect confident detections
# (e.g. Black-capped Chickadee > 0.9, Dark-eyed Junco > 0.8)
curl -sL -o /tmp/soundscape.wav \
  https://raw.githubusercontent.com/birdnet-team/birdnet-test-data/main/soundscape/soundscape.wav
python -m kicau.identify /tmp/soundscape.wav --min-confidence 0.3
```

If the top detections stop looking like confident North-American backyard
birds, the preprocessing or activation math drifted from BirdNET-Analyzer —
fix that before anything else.

## Pull requests

1. Fork, branch, small focused diff.
2. `pytest` green.
3. Describe what you changed and how you verified it (what audio you tested).
