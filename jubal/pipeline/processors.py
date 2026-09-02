"""Pipecat FrameProcessors that drive AgentGateStateMachine off real audio.

Requires the `realtime` extra (`pip install jubal[realtime]`) — this module
is the only place in the codebase that imports pipecat, so the rest of
Jubal (voice-ID gate logic, MCP server, model wrappers) works without it.

Frame flow:

    transport.input() -> InputAudioRawFrame
        -> VoiceGateProcessor   (wake word -> voice ID -> streaming ASR)
            -> AgentUtteranceFrame  -> AgentBridgeProcessor -> SpeakFrame
            -> SpeakFrame (accept chime)                    -> SpeakFrame
        -> TtsSpeakProcessor -> TTSAudioRawFrame -> transport.output()
"""

from __future__ import annotations

import time

import numpy as np
from pipecat.frames.frames import Frame, FrameDirection, InputAudioRawFrame, TTSAudioRawFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor

from jubal.asr.streaming import StreamingRecognizer
from jubal.config import JubalConfig
from jubal.pipeline.agent_bridge import AgentBridge
from jubal.pipeline.state_machine import AgentGateStateMachine, GateAction, GateState
from jubal.tts.synthesizer import Synthesizer
from jubal.voiceid.factory import build_extractor
from jubal.voiceid.gate import GateConfig as VoiceIdGateConfig
from jubal.voiceid.gate import VoiceGate
from jubal.voiceid.store import EmbeddingStore
from jubal.wake.keyword_spotter import WakeWordDetector


class AgentUtteranceFrame(TextFrame):
    """A completed, speaker-verified utterance, ready for the agent."""

    def __init__(self, text: str, speaker: str):
        super().__init__(text=text)
        self.speaker = speaker


class SpeakFrame(TextFrame):
    """Text that should be spoken by TtsSpeakProcessor (agent replies, chimes)."""


def _pcm16_bytes_to_float32(audio: bytes) -> np.ndarray:
    return (np.frombuffer(audio, dtype=np.int16).astype(np.float32)) / 32768.0


class VoiceGateProcessor(FrameProcessor):
    """Owns wake word -> voice ID -> streaming ASR and the state machine
    that sequences them."""

    # How much audio (seconds) to buffer in VERIFYING before extracting an
    # embedding. Short enough to keep latency low, long enough for a usable
    # speaker embedding.
    VERIFICATION_WINDOW_S = 1.5

    def __init__(self, config: JubalConfig, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._wake = WakeWordDetector(config.wake)
        self._extractor = build_extractor(config.voiceid)
        self._gate = VoiceGate(
            EmbeddingStore(config.voiceid.store_path),
            VoiceIdGateConfig(threshold=config.voiceid.threshold, margin=config.voiceid.margin),
        )
        self._asr = StreamingRecognizer(config.streaming_asr)
        self._machine = AgentGateStateMachine(config.gate)
        self._verification_buffer: list[np.ndarray] = []
        self._verification_samples_needed = int(
            self.VERIFICATION_WINDOW_S * config.streaming_asr.sample_rate
        )
        self._active_speaker: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame)
            return  # audio is consumed here, not forwarded raw

        await self.push_frame(frame, direction)

    async def _handle_audio(self, frame: InputAudioRawFrame) -> None:
        now = time.monotonic()
        for action in self._machine.tick(now):
            await self._apply_action(action, now)

        samples = _pcm16_bytes_to_float32(frame.audio)
        sample_rate = frame.sample_rate

        if self._machine.state == GateState.IDLE:
            keyword = self._wake.feed(samples, sample_rate)
            if keyword:
                for action in self._machine.on_wake_word(now):
                    await self._apply_action(action, now)

        elif self._machine.state == GateState.VERIFYING:
            self._verification_buffer.append(samples)
            buffered = sum(len(s) for s in self._verification_buffer)
            if buffered >= self._verification_samples_needed:
                clip = np.concatenate(self._verification_buffer)
                self._verification_buffer = []
                embedding = self._extractor.extract(clip, sample_rate)
                result = self._gate.evaluate(embedding)
                if result.accepted:
                    self._active_speaker = result.speaker
                    for action in self._machine.on_speaker_verified(now):
                        await self._apply_action(action, now)
                else:
                    for action in self._machine.on_speaker_rejected(now):
                        await self._apply_action(action, now)

        elif self._machine.state == GateState.ACTIVE:
            result = self._asr.feed(samples, sample_rate)
            if result.is_endpoint and result.text:
                self._asr.reset()
                for action in self._machine.on_utterance(now):
                    if action == GateAction.FORWARD_UTTERANCE:
                        await self.push_frame(
                            AgentUtteranceFrame(text=result.text, speaker=self._active_speaker),
                            FrameDirection.DOWNSTREAM,
                        )
                    else:
                        await self._apply_action(action, now)
            elif result.is_endpoint:
                self._asr.reset()

    async def _apply_action(self, action: GateAction, now: float) -> None:
        if action == GateAction.SPEAK_ACCEPT:
            await self.push_frame(SpeakFrame(text="Yes?"), FrameDirection.DOWNSTREAM)
        elif action == GateAction.SPEAK_REJECT:
            # Deliberately silent: an unverified voice gets no audible
            # confirmation that a gate exists at all, let alone that it
            # failed. Flip this to a SpeakFrame if you'd rather have an
            # audible "not recognized" for a non-security use case.
            pass
        # START_VOICE_ID / START_ASR / STOP_ASR / RETURN_TO_IDLE need no
        # extra effect here: the state read in _handle_audio already routes
        # audio correctly next frame, and self._asr carries no persistent
        # state that needs explicit start/stop beyond reset().


class AgentBridgeProcessor(FrameProcessor):
    """Forwards a verified utterance to the agent endpoint, speaks the reply."""

    def __init__(self, config: JubalConfig, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AgentUtteranceFrame):
            bridge = AgentBridge(self._config.agent, speaker=frame.speaker)
            reply = await bridge.send(frame.text)
            if reply.text:
                await self.push_frame(SpeakFrame(text=reply.text), FrameDirection.DOWNSTREAM)
            return

        await self.push_frame(frame, direction)


class TtsSpeakProcessor(FrameProcessor):
    """Turns SpeakFrame text into TTSAudioRawFrame audio via sherpa-onnx."""

    def __init__(self, config: JubalConfig, **kwargs):
        super().__init__(**kwargs)
        self._synth = Synthesizer(config.tts)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SpeakFrame):
            audio = self._synth.speak(frame.text)
            pcm16 = (audio.samples * 32767.0).astype(np.int16).tobytes()
            await self.push_frame(
                TTSAudioRawFrame(audio=pcm16, sample_rate=audio.sample_rate, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
            return

        await self.push_frame(frame, direction)
