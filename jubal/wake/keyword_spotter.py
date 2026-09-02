"""Wake word detection via sherpa-onnx's KeywordSpotter.

API per k2-fsa/sherpa-onnx python-api-examples/keyword-spotter.py: build a
KeywordSpotter from tokens/encoder/decoder/joiner + a keywords file, create
one long-lived stream, feed audio incrementally, and reset the stream right
after a detection (the upstream example calls this out explicitly).
"""

from __future__ import annotations

import numpy as np
import sherpa_onnx

from jubal.config import WakeConfig


class WakeWordDetector:
    def __init__(self, config: WakeConfig):
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=f"{config.model_dir}/tokens.txt",
            encoder=f"{config.model_dir}/encoder.onnx",
            decoder=f"{config.model_dir}/decoder.onnx",
            joiner=f"{config.model_dir}/joiner.onnx",
            num_threads=config.num_threads,
            keywords_file=config.keywords_file,
            provider=config.provider,
        )
        self._stream = self._spotter.create_stream()

    def feed(self, samples: np.ndarray, sample_rate: int) -> str | None:
        """Feed a chunk of audio. Returns the detected keyword, or None."""
        self._stream.accept_waveform(sample_rate, samples)
        detected: str | None = None
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                detected = result
                self._spotter.reset_stream(self._stream)
        return detected
