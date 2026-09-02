"""The gate: idle -> verifying -> active -> idle.

This is the one piece of pipeline logic that is genuinely ours (everything
else is glue around sherpa-onnx/Pipecat/MCP), so it's kept free of all of
those: no audio, no I/O, no async. Pipecat FrameProcessors (see
jubal/pipeline/processors.py) drive this with real events and translate its
actions into pipeline effects.
"""

from __future__ import annotations

from enum import Enum, auto

from jubal.config import GateTimingConfig


class GateState(Enum):
    IDLE = auto()
    VERIFYING = auto()
    ACTIVE = auto()


class GateAction(Enum):
    START_VOICE_ID = auto()
    START_ASR = auto()
    STOP_ASR = auto()
    SPEAK_ACCEPT = auto()
    SPEAK_REJECT = auto()
    FORWARD_UTTERANCE = auto()
    RETURN_TO_IDLE = auto()


class AgentGateStateMachine:
    def __init__(self, config: GateTimingConfig | None = None):
        self.config = config or GateTimingConfig()
        self.state = GateState.IDLE
        self._state_entered_at = 0.0

    def _enter(self, state: GateState, now: float) -> None:
        self.state = state
        self._state_entered_at = now

    def on_wake_word(self, now: float) -> list[GateAction]:
        """Wake word heard. Only matters from IDLE — ignored mid-verification
        or mid-conversation so a stray wake-word false-positive can't reset
        an already-active session."""
        if self.state != GateState.IDLE:
            return []
        self._enter(GateState.VERIFYING, now)
        return [GateAction.START_VOICE_ID]

    def on_speaker_verified(self, now: float) -> list[GateAction]:
        if self.state != GateState.VERIFYING:
            return []
        self._enter(GateState.ACTIVE, now)
        return [GateAction.SPEAK_ACCEPT, GateAction.START_ASR]

    def on_speaker_rejected(self, now: float) -> list[GateAction]:
        if self.state != GateState.VERIFYING:
            return []
        self._enter(GateState.IDLE, now)
        return [GateAction.SPEAK_REJECT]

    def on_utterance(self, now: float) -> list[GateAction]:
        """A completed, transcribed utterance arrived. Refreshes the active
        window so a conversation doesn't time out mid-exchange."""
        if self.state != GateState.ACTIVE:
            return []
        self._state_entered_at = now
        return [GateAction.FORWARD_UTTERANCE]

    def tick(self, now: float) -> list[GateAction]:
        """Call periodically (or at least whenever a timeout might have
        elapsed) with the current time. Handles both timeouts without
        needing a real timer/event loop dependency here."""
        elapsed = now - self._state_entered_at
        if self.state == GateState.VERIFYING and elapsed > self.config.verification_timeout_s:
            self._enter(GateState.IDLE, now)
            return [GateAction.SPEAK_REJECT]
        if self.state == GateState.ACTIVE and elapsed > self.config.active_timeout_s:
            self._enter(GateState.IDLE, now)
            return [GateAction.STOP_ASR, GateAction.RETURN_TO_IDLE]
        return []

    def reset(self, now: float) -> None:
        self._enter(GateState.IDLE, now)
