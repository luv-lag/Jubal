"""Small shared audio helpers used across the wake/voiceid/asr/tts wrappers.

Kept dependency-light (numpy + soundfile only) so importing this module
never pulls in sherpa-onnx, Pipecat, or a specific voice-ID backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_wave(path: str | Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Read a wav file as mono float32 samples normalized to [-1, 1].

    Returns (samples, sample_rate). If target_sample_rate is given and
    differs from the file's rate, raises — resampling is the caller's
    responsibility (e.g. record at the model's native rate) rather than a
    silent, potentially lossy implicit conversion here.
    """
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        raise ValueError(
            f"{path}: sample rate {sample_rate} Hz does not match expected "
            f"{target_sample_rate} Hz; re-record or resample before feeding it in."
        )
    return samples, sample_rate


def write_wave(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
