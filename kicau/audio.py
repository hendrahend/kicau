"""Audio loading: decode anything ffmpeg understands to 48 kHz mono float32.

BirdNET expects 3-second chunks at 48 000 Hz. We let ffmpeg do all decoding
and resampling so the app accepts wav/mp3/m4a/ogg/webm (what browsers record).
"""

import subprocess

import numpy as np

SAMPLE_RATE = 48000


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file to mono float32 at target_sr. Raises on failure."""
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", path,
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode()[:300]}")
    sig = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if sig.size == 0:
        raise RuntimeError(f"ffmpeg decoded {path} to zero samples")
    return sig


def chunk_signal(sig: np.ndarray, chunk_s: float = 3.0, sr: int = SAMPLE_RATE):
    """Yield (start_sec, end_sec, chunk) of fixed length; zero-pad the tail.

    Chunks shorter than 1 second are dropped — too little signal to classify.
    """
    chunk_len = int(chunk_s * sr)
    total = sig.shape[0]
    start = 0
    while start < total:
        end = min(start + chunk_len, total)
        chunk = sig[start:end]
        if chunk.shape[0] < sr:  # < 1 s of audio left: not worth classifying
            break
        if chunk.shape[0] < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - chunk.shape[0]))
        yield start / sr, end / sr, chunk
        start += chunk_len
