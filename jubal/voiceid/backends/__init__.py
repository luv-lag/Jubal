"""Embedding extractor backends. Import the specific backend you need directly
(e.g. `from jubal.voiceid.backends.resemblyzer_backend import ResemblyzerEmbeddingExtractor`)
so the optional heavy dependency (sherpa_onnx or resemblyzer) is only
imported when that backend is actually used.
"""

from typing import Protocol

import numpy as np


class EmbeddingExtractor(Protocol):
    """Anything that turns audio samples into a fixed-size speaker embedding."""

    dim: int

    def extract(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        ...


__all__ = ["EmbeddingExtractor"]
