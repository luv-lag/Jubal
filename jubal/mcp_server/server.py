"""MCP server exposing Jubal's building blocks as tools for any MCP client:
enroll a voice, verify a clip against enrolled speakers, transcribe a file,
and synthesize speech — the same primitives the realtime pipeline uses,
without needing a live mic/speaker session.

    python -m jubal.mcp_server.server            # stdio transport
    JUBAL_CONFIG=/path/to/config.yaml python -m jubal.mcp_server.server

Requires the `mcp` extra: `pip install jubal[mcp]`.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from jubal.asr.offline import OfflineTranscriber
from jubal.audio_io import read_wave, write_wave
from jubal.config import JubalConfig
from jubal.tts.synthesizer import Synthesizer
from jubal.voiceid.factory import build_extractor
from jubal.voiceid.gate import GateConfig, VoiceGate
from jubal.voiceid.store import EmbeddingStore

mcp = MCPServer("jubal", instructions="Enroll, verify, transcribe, and speak through Jubal's voice-ID gate.")


@lru_cache(maxsize=1)
def _config() -> JubalConfig:
    path = os.environ.get("JUBAL_CONFIG", "config/default.yaml")
    return JubalConfig.load(path)


@lru_cache(maxsize=1)
def _store() -> EmbeddingStore:
    return EmbeddingStore(_config().voiceid.store_path)


@lru_cache(maxsize=1)
def _extractor():
    return build_extractor(_config().voiceid)


@lru_cache(maxsize=1)
def _gate() -> VoiceGate:
    cfg = _config().voiceid
    return VoiceGate(_store(), GateConfig(threshold=cfg.threshold, margin=cfg.margin))


@lru_cache(maxsize=1)
def _transcriber() -> OfflineTranscriber:
    return OfflineTranscriber(_config().offline_asr)


@lru_cache(maxsize=1)
def _synth() -> Synthesizer:
    return Synthesizer(_config().tts)


@mcp.tool()
def enroll_voice(name: str, wav_path: str) -> str:
    """Enroll a speaker from a wav clip of their voice (a few seconds, 16kHz mono is safest)."""
    samples, sample_rate = read_wave(wav_path)
    embedding = _extractor().extract(samples, sample_rate)
    _store().enroll(name, embedding)
    _store().save()
    return f"enrolled {name!r} ({len(_store().embeddings_for(name))} sample(s) on file)"


@mcp.tool()
def list_enrolled_speakers() -> list[str]:
    """List speaker names currently enrolled in the voice-ID store."""
    return _store().speakers()


@mcp.tool()
def verify_speaker(wav_path: str) -> dict:
    """Check a wav clip against enrolled speakers. Returns accepted/speaker/score."""
    samples, sample_rate = read_wave(wav_path)
    embedding = _extractor().extract(samples, sample_rate)
    result = _gate().evaluate(embedding)
    return {
        "accepted": result.accepted,
        "speaker": result.speaker,
        "score": result.score,
        "runner_up_score": result.runner_up_score,
        "reason": result.reason,
    }


@mcp.tool()
def transcribe(wav_path: str) -> str:
    """Transcribe a wav file to text using the offline ASR model."""
    samples, sample_rate = read_wave(wav_path)
    return _transcriber().transcribe(samples, sample_rate)


@mcp.tool()
def speak(text: str) -> str:
    """Synthesize speech for `text` and return the path to the generated wav file."""
    audio = _synth().speak(text)
    out_path = Path(tempfile.gettempdir()) / f"jubal-tts-{abs(hash(text))}.wav"
    write_wave(out_path, audio.samples, audio.sample_rate)
    return str(out_path)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
