"""One-time download of the BirdNET v2.4 open weights (fp16 TFLite, ~53 MB).

Source: Zenodo record 15050749 (BirdNET team).
License of the weights: CC-BY-NC-4.0 (free for research/education/personal use).
The app code itself is MIT; only the weights carry the NC term.
"""

import os
import urllib.request
import zipfile

ZENODO_URL = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_fp16.zip?download=1"
MODEL_FILE = "audio-model-fp16.tflite"
META_FILE = "meta-model.tflite"  # location filter: (lat, lon, week) -> per-species probability


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base, "model")
    os.makedirs(model_dir, exist_ok=True)
    dest = os.path.join(model_dir, MODEL_FILE)
    meta_dest = os.path.join(model_dir, META_FILE)
    if os.path.exists(dest) and os.path.exists(meta_dest):
        print(f"Models already present in {model_dir}")
        return
    zip_path = os.path.join(model_dir, "birdnet_v24_fp16.zip")
    print("Downloading BirdNET v2.4 weights (~53 MB)...")
    urllib.request.urlretrieve(ZENODO_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(MODEL_FILE, model_dir)
        z.extract("meta-model.tflite", model_dir)
        os.rename(os.path.join(model_dir, "meta-model.tflite"), meta_dest)
    os.remove(zip_path)
    print(f"Saved to {dest} and {meta_dest}")


if __name__ == "__main__":
    main()
