"""Build the configured embedding extractor without importing every backend eagerly."""

from __future__ import annotations

from jubal.config import VoiceIdConfig
from jubal.voiceid.backends import EmbeddingExtractor


def build_extractor(config: VoiceIdConfig) -> EmbeddingExtractor:
    if config.backend == "resemblyzer":
        from jubal.voiceid.backends.resemblyzer_backend import ResemblyzerEmbeddingExtractor

        return ResemblyzerEmbeddingExtractor()
    if config.backend == "sherpa":
        from jubal.voiceid.backends.sherpa_backend import SherpaEmbeddingExtractor

        if not config.sherpa_model:
            raise ValueError("voiceid.sherpa_model must be set to use the sherpa backend")
        return SherpaEmbeddingExtractor(
            model=config.sherpa_model,
            num_threads=config.num_threads,
            provider=config.provider,
        )
    raise ValueError(f"unknown voiceid backend: {config.backend!r}")
