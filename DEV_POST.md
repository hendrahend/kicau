# DEV POST — copy everything below the line into dev.to via the Week 1 challenge page
# Title:
# The birds outside my window have Indonesian names. My bird app didn't.
# Replace <your-username> with your GitHub username. Add your screenshots
# where marked. Fill your GitHub repo link in ## Code.

---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)*

I live in Jakarta. The birds here don't sing in English — but every bird-identifying app I tried did. Six apps, and not one could tell me what the bird outside my kos is called in the language I actually speak.

So I built Kicau: an offline bird-call identifier that speaks Indonesian.

## What I Built

**Kicau** records bird audio — from your microphone or an uploaded file — and identifies the species, on-device, in English or Indonesian. No internet, no server, no account. It works in the middle of a national park with zero bars of signal, and your recordings never leave your device. An EN/ID toggle in the corner switches the whole UI *and* the species names — *Orthotomus sutorius* reads "Common Tailorbird" or "Cinenen pisang" depending on the toggle.

It's for Indonesian birders, hikers, and anyone who's ever heard a bird on a trail and wondered what it was. The screen is the shortest part of the experience: record ten seconds, read the result, pocket the phone, go find the bird. Tap "Use my location" and it also weighs which birds actually live near you.

Yes — there are other bird identifiers in this challenge. Kicau is the one built for Indonesia's 280 million people: Bahasa Indonesia UI, Indonesian species names ("Cikrak bambu", not just *Abroscopus superciliaris*), two pip packages, one HTML file, zero build step.

## Demo

Kicau is offline-first, so there's no hosted demo — that would defeat the point. It runs on your machine:

```bash
git clone https://github.com/<your-username>/kicau && cd kicau
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py   # one-time, ~53 MB
python -m kicau.server             # open http://127.0.0.1:8765
```

![Kicau main UI — record and upload tabs](https://raw.githubusercontent.com/<your-username>/kicau/main/docs/demo-ui.png)

A Common Tailorbird field recording, identified on-device:

![Kicau result — species cards with confidence bars and the per-3-second detection timeline](https://raw.githubusercontent.com/<your-username>/kicau/main/docs/demo-result.png)

Or from the terminal:

```bash
python -m kicau.identify recording.mp3 --top-n 3
```

## Code

<!-- Replace with your repo link, then delete this comment -->
https://github.com/<your-username>/kicau

MIT licensed. What's inside:

| Path | What |
|---|---|
| `kicau/identify.py` | Model loading, chunking, inference, Indonesian labels |
| `kicau/audio.py` | ffmpeg-based decode/resample to 48 kHz mono |
| `kicau/server.py` | stdlib-only HTTP server + `/api/identify` |
| `kicau/labels_id.txt` | 6,522 species labels in Indonesian (BirdNET team) |
| `web/index.html` | Single-file offline UI — record, upload, timeline, playback |
| `tests/` | Pipeline tests, including a real-audio classification test |

## How I Built It

The whole experience is built around one open-weight model:

| Piece | Role | License |
|---|---|---|
| **BirdNET v2.4** (Cornell Lab of Ornithology / TU Chemnitz) | 6,522-species bioacoustics classifier; I use the fp16 TFLite variant (~26 MB) | CC BY-NC 4.0 (weights) |
| **LiteRT** (`ai-edge-litert`) | On-device inference runtime — no TensorFlow install, no GPU | Apache 2.0 |
| **BirdNET's Indonesian labels** | Species names in Bahasa Indonesia, shipped by the BirdNET team | MIT (analyzer repo) |

One honest caveat on the Indonesian names: the BirdNET team's Indonesian label set covers 1,227 of the 6,522 species so far — the rest fall back to English while the translation grows. The covered ones include the birds you're most likely to meet in Indonesia (Cinenen, Cikrak, Elang-alap).

Pipeline:

```
mic / upload → ffmpeg (→ 48 kHz mono) → 3-second chunks →
BirdNET v2.4 TFLite (LiteRT, on-device) → flat sigmoid →
top species per chunk, Indonesian names
```

I verified the math against BirdNET-Analyzer's source line by line — chunk length, sample rate, sigmoid sensitivity — so results match the reference implementation. Then I ran BirdNET's own reference soundscape through it: Black-capped Chickadee 0.92, Dark-eyed Junco 0.88, House Finch 0.83. And on an openly-licensed field recording of a Common Tailorbird (Vis M via iNaturalist, CC BY-SA 4.0, Wikimedia Commons): **"Common Tailorbird (*Orthotomus sutorius*)" at 1.00 confidence** — correct species. (Flip the app's EN/ID toggle and it reads "Cinenen pisang" — species names switch languages too.)

One honest miss, because that's part of the deal: on a noisy caged-bird training clip, confidence stayed low (0.06–0.21) and the top guess was wrong. Clean field recordings are this model's home turf; noisy clips aren't.

**Location-aware, too.** BirdNET ships a second tiny model that takes your latitude, longitude, and the week of the year, and returns how likely each species is there right now. Tap "Use my location" and Kicau multiplies the acoustic confidence by that local likelihood — a bird that merely sounds similar but doesn't live here gets demoted. I verified it: in Jakarta in October the top picks are Burung-gereja erasia (Eurasian Tree Sparrow) and Perkutut jawa (Zebra Dove) — exactly the birds outside my window — while Berlin returns Great Tit and Eurasian Blackbird.

## Why Does Open Innovation Matter?

1. **No signal, no problem.** The entire point is identifying birds *outside*. A cloud API is useless in a national park. A local open-weight model is the only architecture that works where the birds actually are.
2. **Your recordings stay yours.** Audio is decoded, chunked, classified, and deleted — on your machine. There is nowhere to upload it to, because there is no server.
3. **It costs nothing, forever.** No per-request pricing, no API key to babysit. A school birding club can run it on one laptop for years.
4. **I could actually verify it.** I checked my preprocessing against the reference implementation line by line and tested on known audio. Try getting that level of verification from a black-box API.

One honest license note: the BirdNET *weights* are CC BY-NC 4.0 (free for research, education, and personal use) while the analyzer code is MIT and my code is MIT. Open weights with a non-commercial term — worth knowing, and fine for this project.

## Prize Categories

- **Best Use of GitHub Copilot** (GitHub, $100): the repo ships a GitHub Actions CI workflow that installs dependencies, downloads the BirdNET weights, and runs the full pytest suite — including a real-audio classification test — on every push.

---

*Built for Hacktoberfest 2026. Then go outside. That's the whole point.* 🌿
