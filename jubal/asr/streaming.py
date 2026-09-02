"""Streaming ASR via sherpa-onnx's OnlineRecognizer, for the live pipeline.

API per k2-fsa/sherpa-onnx python-api-examples/speech-recognition-from-
microphone-with-endpoint-detection.py: OnlineRecognizer.from_transducer(...)
with endpoint-detection rules enabled, one stream fed incrementally, decode
while is_ready(), check is_endpoint() to know when an utterance is done.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sherpa_onnx

from jubal.config import StreamingAsrConfig


@dataclass
class PartialResult:
    text: str
    is_endpoint: bool


class StreamingRecognizer:
    def __init__(self, config: StreamingAsrConfig):
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=config.tokens,
            encoder=config.encoder,
            decoder=config.decoder,
            joiner=config.joiner,
            num_threads=config.num_threads,
            sample_rate=config.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
            decoding_method="greedy_search",
            provider=config.provider,
        )
        self._stream = self._recognizer.create_stream()

    def feed(self, samples: np.ndarray, sample_rate: int) -> PartialResult:
        """Feed a chunk of audio, return the current partial text and whether
        an utterance endpoint was just detected. Caller should treat the
        endpoint as "utterance complete" and call reset() afterwards.
        """
        self._stream.accept_waveform(sample_rate, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._recognizer.get_result(self._stream)
        endpoint = self._recognizer.is_endpoint(self._stream)
        return PartialResult(text=text, is_endpoint=endpoint)

    def reset(self) -> None:
        self._recognizer.reset(self._stream)
