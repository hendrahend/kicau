# Security Policy

## What this app does with your data

Kicau processes audio **entirely on your device**. Uploaded files are written
to a temporary file, analyzed locally, and deleted immediately after. No
audio, results, or telemetry are sent anywhere.

## Reporting a vulnerability

Open a GitHub issue (or email the repo owner) with:

- what you found and where (`file:line` if you can),
- steps to reproduce,
- what you think the impact is.

Please don't open a public issue with an exploit that puts users at risk
before a fix exists — give the maintainer a few days to respond first.

## Scope notes

- The BirdNET v2.4 weights are third-party (CC-BY-NC-4.0); model-weight
  vulnerabilities are upstream's domain.
- `ffmpeg` is a system dependency; keep yours updated.
