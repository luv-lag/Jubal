"""Fast-prototype voice-ID backend: Resemblyzer, no ONNX export needed.

Good for tuning GateConfig thresholds/margins before committing to the
sherpa-onnx/3D-Speaker backend for production. Implements the same
EmbeddingExtractor protocol, so gate.py and everything above it is
unaffected by which backend is in use.
"""

from __future__ import annotations

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

RESEMBLYZER_SAMPLE_RATE = 16000
EMBEDDING_DIM = 256


class ResemblyzerEmbeddingExtractor:
    def __init__(self):
        self._encoder = VoiceEncoder()
        self.dim = EMBEDDING_DIM

    def extract(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = preprocess_wav(samples, source_sr=sample_rate)
        embedding = self._encoder.embed_utterance(wav)
        return np.asarray(embedding, dtype=np.float32)
