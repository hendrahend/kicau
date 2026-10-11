"""Bird-call identification with BirdNET v2.4 running locally via LiteRT.

Pipeline (mirrors BirdNET-Analyzer v2.4 defaults):
  audio file -> 48 kHz mono -> 3 s chunks -> TFLite fp16 model
  -> flat sigmoid (sensitivity 1.0) -> top-N species per chunk

Species labels come in English and Indonesian (both shipped by the BirdNET team);
`identify_file(..., lang="en"|"id")` picks which set to report.
"""

import os

import numpy as np

from .audio import SAMPLE_RATE, chunk_signal, load_audio

MODEL_INPUT_SAMPLES = 3 * SAMPLE_RATE  # 144000
NUM_SPECIES = 6522


def flat_sigmoid(x: np.ndarray, sensitivity: float = -1.0, bias: float = 1.0) -> np.ndarray:
    """BirdNET-Analyzer's activation: 1 / (1 + exp(sensitivity * clip(x + bias)))."""
    return 1.0 / (1.0 + np.exp(sensitivity * np.clip(x + bias, -20.0, 20.0)))


def load_labels(path: str):
    """Parse 'Scientific name_Common name' lines -> [(scientific, common)]."""
    labels = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "_" in line:
                sci, common = line.split("_", 1)
            else:
                sci, common = line, line
            labels.append((sci.strip(), common.strip()))
    return labels


class BirdIdentifier:
    def __init__(self, model_path: str, labels_path: str):
        from ai_edge_litert.interpreter import Interpreter

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\nRun: python scripts/download_model.py"
            )
        self.interpreter = Interpreter(model_path=model_path, num_threads=2)
        self.interpreter.allocate_tensors()
        details = self.interpreter.get_input_details()[0]
        self.input_index = details["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        assert tuple(details["shape"]) == (1, MODEL_INPUT_SAMPLES), (
            f"unexpected model input shape {details['shape']}"
        )
        self.labels = load_labels(labels_path)
        assert len(self.labels) == NUM_SPECIES, (
            f"expected {NUM_SPECIES} labels, got {len(self.labels)}"
        )
        # Location filter model sits next to the audio model; loaded lazily.
        self._meta_path = os.path.join(
            os.path.dirname(os.path.abspath(model_path)), "meta-model.tflite"
        )
        self._loc_filter = None
        # Load the sibling language set when present (labels_id.txt <-> labels_en.txt).
        self.label_sets = {}
        label_dir = os.path.dirname(os.path.abspath(labels_path))
        for lang, fname in (("id", "labels_id.txt"), ("en", "labels_en.txt")):
            p = os.path.join(label_dir, fname)
            if os.path.exists(p):
                self.label_sets[lang] = load_labels(p)
        self.default_lang = "en" if "en" in self.label_sets else "id"

    def predict_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Return confidence per species (0..1) for one 3 s chunk."""
        x = np.asarray(chunk, dtype=np.float32).reshape(1, MODEL_INPUT_SAMPLES)
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_index)[0]
        return flat_sigmoid(logits).astype(np.float64)

    def location_filter(self):
        """Lazily load BirdNET's location model (lat, lon, week -> species probs)."""
        if self._loc_filter is None:
            from .location import LocationFilter

            self._loc_filter = LocationFilter(self._meta_path)
        return self._loc_filter

    def identify_file(self, path: str, top_n: int = 3, min_confidence: float = 0.1,
                      lang: str = "en", lat: float | None = None,
                      lon: float | None = None, week: int | None = None):
        """Identify birds in an audio file. Returns per-chunk detections.

        lang picks the species-name language: "en" (default) or "id".
        When lat/lon are given, each acoustic confidence is multiplied by the
        species' local occurrence probability, so birds that merely sound
        similar but don't live here get demoted.
        """
        labels = self.label_sets.get(lang, self.label_sets.get(self.default_lang,
                                                                self.labels))
        loc_probs = None
        if lat is not None and lon is not None:
            loc_probs = self.location_filter().predict(lat, lon, week)
        sig = load_audio(path)
        chunks = []
        for start, end, chunk in chunk_signal(sig):
            conf = self.predict_chunk(chunk)
            if loc_probs is not None:
                conf = conf * loc_probs
            top = np.argsort(conf)[::-1][:top_n]
            preds = []
            for i in top:
                c = float(conf[i])
                if c < min_confidence:
                    continue
                sci, common = labels[int(i)]
                preds.append(
                    {
                        "scientific": sci,
                        "common": common,
                        "confidence": round(c, 4),
                    }
                )
            chunks.append({"start": round(start, 2), "end": round(end, 2), "predictions": preds})
        return {
            "file": os.path.basename(path),
            "duration": round(sig.shape[0] / SAMPLE_RATE, 2),
            "chunks": chunks,
            "summary": summarize(chunks),
            "location": {"lat": lat, "lon": lon} if loc_probs is not None else None,
        }

    def close(self):
        self.interpreter = None


def summarize(chunks):
    """Best confidence per species across all chunks, sorted descending."""
    best = {}
    for ch in chunks:
        for p in ch["predictions"]:
            key = (p["scientific"], p["common"])
            if key not in best or p["confidence"] > best[key]["confidence"]:
                best[key] = {
                    "scientific": p["scientific"],
                    "common": p["common"],
                    "confidence": p["confidence"],
                    "at": ch["start"],
                }
    return sorted(best.values(), key=lambda d: d["confidence"], reverse=True)


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Identify bird calls in an audio file (offline).")
    ap.add_argument("audio", help="audio file (wav/mp3/m4a/ogg/webm)")
    ap.add_argument("--model", default="model/audio-model-fp16.tflite")
    ap.add_argument("--labels", default="kicau/labels_id.txt")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--min-confidence", type=float, default=0.1)
    ap.add_argument("--lang", default="en", choices=("en", "id"),
                    help="species-name language")
    ap.add_argument("--lat", type=float, default=None, help="recording latitude")
    ap.add_argument("--lon", type=float, default=None, help="recording longitude")
    args = ap.parse_args()

    ident = BirdIdentifier(args.model, args.labels)
    try:
        print(json.dumps(ident.identify_file(args.audio, args.top_n, args.min_confidence,
                                             lang=args.lang, lat=args.lat, lon=args.lon),
                         ensure_ascii=False, indent=2))
    finally:
        ident.close()


if __name__ == "__main__":
    main()
