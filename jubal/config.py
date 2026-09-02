"""Typed config for Jubal, loaded from a YAML file (see config/default.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WakeConfig:
    model_dir: str = ""
    keywords_file: str = "keywords.txt"
    num_threads: int = 1
    provider: str = "cpu"


@dataclass
class VoiceIdConfig:
    backend: str = "resemblyzer"  # "resemblyzer" | "sherpa"
    sherpa_model: str = ""
    store_path: str = "voice_profiles.json"
    threshold: float = 0.72
    margin: float = 0.05
    num_threads: int = 1
    provider: str = "cpu"


@dataclass
class StreamingAsrConfig:
    tokens: str = ""
    encoder: str = ""
    decoder: str = ""
    joiner: str = ""
    sample_rate: int = 16000
    num_threads: int = 1
    provider: str = "cpu"


@dataclass
class OfflineAsrConfig:
    tokens: str = ""
    encoder: str = ""
    decoder: str = ""
    joiner: str = ""
    num_threads: int = 1
    provider: str = "cpu"


@dataclass
class TtsConfig:
    # VITS (e.g. a vits-piper-* voice) is the only backend Synthesizer
    # implements today. sherpa-onnx also supports Kokoro/Matcha/Kitten
    # offline TTS models via the same OfflineTtsConfig shape; add a branch
    # in jubal/tts/synthesizer.py if you need one of those.
    model_dir: str = ""
    speaker_id: int = 0
    speed: float = 1.0
    num_threads: int = 1
    provider: str = "cpu"


@dataclass
class GateTimingConfig:
    verification_timeout_s: float = 2.5
    active_timeout_s: float = 8.0


@dataclass
class AgentConfig:
    endpoint_url: str = "http://localhost:8000/agent"
    timeout_s: float = 15.0


@dataclass
class JubalConfig:
    wake: WakeConfig = field(default_factory=WakeConfig)
    voiceid: VoiceIdConfig = field(default_factory=VoiceIdConfig)
    streaming_asr: StreamingAsrConfig = field(default_factory=StreamingAsrConfig)
    offline_asr: OfflineAsrConfig = field(default_factory=OfflineAsrConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    gate: GateTimingConfig = field(default_factory=GateTimingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def load(cls, path: str | Path) -> "JubalConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JubalConfig":
        return cls(
            wake=WakeConfig(**raw.get("wake", {})),
            voiceid=VoiceIdConfig(**raw.get("voiceid", {})),
            streaming_asr=StreamingAsrConfig(**raw.get("streaming_asr", {})),
            offline_asr=OfflineAsrConfig(**raw.get("offline_asr", {})),
            tts=TtsConfig(**raw.get("tts", {})),
            gate=GateTimingConfig(**raw.get("gate", {})),
            agent=AgentConfig(**raw.get("agent", {})),
        )
