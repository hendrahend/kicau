"""Location-aware species filtering with BirdNET's meta model.

The meta model takes (latitude, longitude, week-of-year) and returns the
probability of each of the 6522 species occurring there. Multiplying the
acoustic confidence by this local likelihood demotes birds that merely
sound similar but don't live here — the same idea as eBird-based filtering,
running fully on-device.
"""

import datetime
import os

import numpy as np


class LocationFilter:
    def __init__(self, model_path: str):
        from ai_edge_litert.interpreter import Interpreter

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Location model not found: {model_path}\n"
                "Run: python scripts/download_model.py"
            )
        self.interpreter = Interpreter(model_path=model_path, num_threads=2)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    @staticmethod
    def current_week() -> int:
        # BirdNET uses weeks 1..48; clamp the ISO week into range.
        return min(48, max(1, datetime.date.today().isocalendar()[1]))

    def predict(self, lat: float, lon: float, week: int | None = None) -> np.ndarray:
        """Per-species occurrence probability (0..1) for a location."""
        if week is None:
            week = self.current_week()
        sample = np.array([[lat, lon, week]], dtype=np.float32)
        self.interpreter.set_tensor(self.input_index, sample)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0].astype(np.float64)
