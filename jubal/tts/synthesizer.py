"""Speech synthesis via sherpa-onnx's OfflineTts (VITS backend).

API per k2-fsa/sherpa-onnx python-api-examples/offline-tts.py: build an
OfflineTtsConfig wrapping an OfflineTtsVitsModelConfig, validate it,
construct OfflineTts, then generate() with a GenerationConfig (speaker id +
speed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from jubal.config import TtsConfig


@dataclass
class SynthesizedAudio:
    samples: np.ndarray
    sample_rate: int


class Synthesizer:
    def __init__(self, config: TtsConfig):
        model_dir = Path(config.model_dir)
        espeak_data_dir = model_dir / "espeak-ng-data"
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_dir / "model.onnx"),
                    tokens=str(model_dir / "tokens.txt"),
                    data_dir=str(espeak_data_dir) if espeak_data_dir.exists() else "",
                ),
                provider=config.provider,
                num_threads=config.num_threads,
            ),
        )
        if not tts_config.validate():
            raise ValueError(f"invalid sherpa-onnx TTS config for model_dir={model_dir}")
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._config = config

    def speak(self, text: str) -> SynthesizedAudio:
        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.sid = self._config.speaker_id
        gen_config.speed = self._config.speed
        audio = self._tts.generate(text, gen_config)
        return SynthesizedAudio(
            samples=np.asarray(audio.samples, dtype=np.float32),
            sample_rate=audio.sample_rate,
        )
