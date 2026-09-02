"""File-based ASR via sherpa-onnx's OfflineRecognizer, used by the MCP
`transcribe` tool where a whole clip is available up front.

API per k2-fsa/sherpa-onnx python-api-examples/offline-decode-files.py:
build via OfflineRecognizer.from_transducer(...) (or from_whisper(...)),
create a stream per clip, accept_waveform once, decode_streams(), read
stream.result.text.
"""

from __future__ import annotations

import numpy as np
import sherpa_onnx

from jubal.config import OfflineAsrConfig


class OfflineTranscriber:
    def __init__(self, config: OfflineAsrConfig):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=config.tokens,
            encoder=config.encoder,
            decoder=config.decoder,
            joiner=config.joiner,
            num_threads=config.num_threads,
            provider=config.provider,
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_streams([stream])
        return stream.result.text
