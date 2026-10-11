---
title: "Kicau — a bird-call identifier that works with zero bars of signal"
published: false
tags: devchallenge, hf26challenge
ai_disclosure_level: some_ai
---

*This is a submission for the Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass*

# Kicau — a bird-call identifier that works with zero bars of signal

**Repo:** https://github.com/<your-username>/kicau (MIT)
**Demo:** `python -m kicau.server` → open http://127.0.0.1:8765, hit record.

## What I Built

Kicau identifies bird calls from a phone or laptop microphone — and the
entire thing runs on-device. Record 10 seconds of birdsong on a trail with
no signal, and it still tells you what you're hearing, in Indonesian.

It's built for Indonesian birders specifically: the UI and instructions are
Bahasa Indonesia, and species names use the Indonesian common names the
BirdNET team ships ("Cikrak bambu", not just *Abroscopus superciliaris*).
Indonesia sits in the most bird-diverse region on earth, and every other
identifier in this challenge speaks English first.

The screen is the shortest part of the experience: record, read the result,
pocket the phone, keep walking.

## Demo

> ✏️ TODO: after the field test, add one of these:
> - a 30-second screen recording of a real identification, or
> - 2–3 photos from outside with the results screenshot.
>
> The challenge rewards this explicitly — "bonus points if you take it
> outside, use it, and tell us how it went."

## Code

https://github.com/<your-username>/kicau — MIT, created fresh for this
challenge. Contains:

- `kicau/identify.py` — BirdNET v2.4 inference via LiteRT (verified against
  BirdNET-Analyzer's source, line by line)
- `kicau/audio.py` — decode → 48 kHz mono → 3-second chunks
- `kicau/server.py` — stdlib-only HTTP server, no framework
- `web/index.html` — single-file UI, zero CDN dependencies (works offline
  after first load): SVG icons, per-chunk detection timeline, audio playback
- `scripts/download_model.py` — one-time 26 MB model download; airplane mode
  after that
- `tests/` (pytest, 4/4 green), `.github/workflows/ci.yml`, CONTRIBUTING,
  CODE_OF_CONDUCT, SECURITY

Only two runtime dependencies: `numpy` + `ai-edge-litert`.

## How I Built It

```
mic / upload → ffmpeg (→ 48 kHz mono) → 3-second chunks →
BirdNET v2.4 TFLite (LiteRT, on-device) → flat sigmoid →
top species per chunk, Indonesian names
```

The open pieces are what make it work:

- **BirdNET v2.4 open weights** (Cornell Lab of Ornithology / TU Chemnitz,
  6,522 species) do the actual identification. Not an API — a 26 MB file
  that lives next to the code.
- **LiteRT** runs the model locally. No TensorFlow install, no GPU, no
  server. A five-year-old laptop handles it.
- **BirdNET's Indonesian species labels**, shipped by the BirdNET team.
- Everything I wrote is **MIT**.

Honest limits: I ran BirdNET's own reference soundscape through it and got
confident, correct detections (Black-capped Chickadee 0.92, Dark-eyed Junco
0.88, House Finch 0.83). On a random low-quality caged-bird clip, confidence
dropped to 0.06–0.21 — the model shines on clean field recordings, which is
exactly what the field test below is for. Also worth knowing: the BirdNET
*weights* are CC-BY-NC-4.0 (free for research, education, personal use);
the analyzer code and my code are MIT.

"Aren't there already six bird identifiers in this challenge?" Probably —
the prompt itself suggested the idea. Kicau isn't competing on the idea.
Merlin Bird ID is closed-source, needs an account, and uploads your
recordings to a server. BirdNET-Analyzer is a heavy English-only research
workbench. Kicau is MIT, fully offline, two pip packages, and made to be
opened on a phone at 6 AM on an Indonesian trail.

## Why Does Open Innovation Matter?

1. **No signal, no problem.** The whole point is identifying birds
   *outside*. A cloud API is useless in a national park. A local
   open-weight model is the only architecture that works there.
2. **Your recordings stay yours.** Audio is decoded, chunked, classified,
   and deleted — on your machine. Nothing is uploaded, because there is
   nowhere to upload it to.
3. **It costs nothing, forever.** No per-request pricing, no API key to
   babysit. A school birding club can run it on one laptop for years.
4. **I could actually verify it.** I checked my preprocessing and activation
   math against BirdNET-Analyzer's source line by line, then ran BirdNET's
   own reference soundscape through it. Try doing that level of verification
   against a black-box API.

## Field test

> ✏️ TODO: take your laptop/phone outside at dawn (birds are most vocal
> 05:30–06:30), record real birdsong, and write 2–3 sentences here about
> where you went and what it identified. Real field results beat every
> architecture paragraph above.

## Try it

```bash
git clone https://github.com/<your-username>/kicau && cd kicau
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py   # one-time; afterwards, airplane mode is fine
python -m kicau.server
```

Then go outside. That's the whole point. 🌿

## Prize Categories

- **Overall (Best in Show)** — Kicau uses no partner technology, so this is
  the category it genuinely enters.

---

*Disclosure: I built Kicau with Muse (Meta's AI assistant) as a pair
programmer — it wrote the inference pipeline and this post draft. I reviewed
and edited everything, and the verification numbers above come from the
tests in the repo.*
