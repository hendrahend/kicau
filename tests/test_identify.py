"""Smoke tests for the Kicau pipeline. Needs the model downloaded (see README)."""

import os
import wave

import numpy as np
import pytest

from kicau.audio import chunk_signal
from kicau.identify import BirdIdentifier, flat_sigmoid, load_labels

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE, "model", "audio-model-fp16.tflite")
LABELS = os.path.join(BASE, "kicau", "labels_id.txt")

requires_model = pytest.mark.skipif(not os.path.exists(MODEL), reason="model not downloaded")


def make_chirp_wav(path, seconds=6.0, sr=48000):
    """Synthetic bird-like chirp: sine sweep 2->6 kHz, pulsed. Not a real bird,
    just exercises the full decode -> chunk -> inference path."""
    t = np.arange(int(seconds * sr)) / sr
    f = 2000 + 4000 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))  # warbling sweep
    sig = 0.5 * np.sin(2 * np.pi * np.cumsum(f) / sr)
    sig *= (0.5 + 0.5 * np.sin(2 * np.pi * 8 * t)) ** 2  # amplitude pulses
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def test_flat_sigmoid_silence_is_low():
    # silence (logit ~ -5.5) must map to a tiny confidence
    assert flat_sigmoid(np.array([-5.5]))[0] < 0.05
    # strong logit must map near 1
    assert flat_sigmoid(np.array([5.0]))[0] > 0.9


def test_labels_count():
    labels = load_labels(LABELS)
    assert len(labels) == 6522
    sci, common = labels[0]
    assert sci and common


def test_chunking():
    sig = np.zeros(48000 * 7, dtype=np.float32)
    chunks = list(chunk_signal(sig))
    assert len(chunks) == 3  # 3 s + 3 s + 1 s tail (tail zero-padded, >= 1 s kept)
    for _, _, c in chunks:
        assert c.shape == (144000,)


@requires_model
def test_full_pipeline(tmp_path):
    wav = str(tmp_path / "chirp.wav")
    make_chirp_wav(wav)
    ident = BirdIdentifier(MODEL, LABELS)
    try:
        result = ident.identify_file(wav)
        assert result["duration"] == pytest.approx(6.0, abs=0.1)
        assert len(result["chunks"]) == 2
        assert isinstance(result["summary"], list)
        for s in result["summary"]:
            assert 0.0 <= s["confidence"] <= 1.0
            assert s["scientific"] and s["common"]
    finally:
        ident.close()
