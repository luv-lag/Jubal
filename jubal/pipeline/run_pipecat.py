"""Assemble and run the realtime pipeline against local mic/speaker audio.

    python -m jubal.pipeline.run_pipecat --config config/default.yaml

Requires the `realtime` extra: `pip install jubal[realtime]`.
"""

from __future__ import annotations

import argparse
import asyncio

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from jubal.config import JubalConfig
from jubal.pipeline.processors import AgentBridgeProcessor, TtsSpeakProcessor, VoiceGateProcessor


async def run(config: JubalConfig) -> None:
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=config.streaming_asr.sample_rate,
        )
    )

    pipeline = Pipeline(
        [
            transport.input(),
            VoiceGateProcessor(config),
            AgentBridgeProcessor(config),
            TtsSpeakProcessor(config),
            transport.output(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams())
    runner = PipelineRunner()
    await runner.run(task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JubalConfig YAML file")
    args = parser.parse_args()

    config = JubalConfig.load(args.config)
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
