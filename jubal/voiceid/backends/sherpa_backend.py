"""3D-Speaker / WeSpeaker embeddings via sherpa-onnx's SpeakerEmbeddingExtractor.

API per k2-fsa/sherpa-onnx python-api-examples/speaker-identification.py:
build a SpeakerEmbeddingExtractorConfig, create a stream, feed the whole
utterance, call extractor.compute(stream) once input_finished() is set.
"""

from __future__ import annotations

import numpy as np
import sherpa_onnx


class SherpaEmbeddingExtractor:
    def __init__(self, model: str, num_threads: int = 1, provider: str = "cpu", debug: bool = False):
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim = self._extractor.dim

    def extract(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()
        embedding = self._extractor.compute(stream)
        return np.asarray(embedding, dtype=np.float32)
