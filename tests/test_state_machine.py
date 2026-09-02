from jubal.config import GateTimingConfig
from jubal.pipeline.state_machine import AgentGateStateMachine, GateAction, GateState


def make_machine(**overrides):
    return AgentGateStateMachine(GateTimingConfig(**overrides))


def test_starts_idle():
    m = make_machine()
    assert m.state == GateState.IDLE


def test_wake_word_moves_to_verifying_and_starts_voice_id():
    m = make_machine()
    actions = m.on_wake_word(now=0.0)
    assert m.state == GateState.VERIFYING
    assert actions == [GateAction.START_VOICE_ID]


def test_wake_word_ignored_while_not_idle():
    m = make_machine()
    m.on_wake_word(now=0.0)
    actions = m.on_wake_word(now=0.1)  # already VERIFYING
    assert actions == []
    assert m.state == GateState.VERIFYING


def test_speaker_verified_moves_to_active_and_starts_asr():
    m = make_machine()
    m.on_wake_word(now=0.0)
    actions = m.on_speaker_verified(now=0.5)
    assert m.state == GateState.ACTIVE
    assert GateAction.SPEAK_ACCEPT in actions
    assert GateAction.START_ASR in actions


def test_speaker_rejected_returns_to_idle():
    m = make_machine()
    m.on_wake_word(now=0.0)
    actions = m.on_speaker_rejected(now=0.5)
    assert m.state == GateState.IDLE
    assert actions == [GateAction.SPEAK_REJECT]


def test_verification_timeout_returns_to_idle():
    m = make_machine(verification_timeout_s=2.0)
    m.on_wake_word(now=0.0)
    actions = m.tick(now=2.1)
    assert m.state == GateState.IDLE
    assert actions == [GateAction.SPEAK_REJECT]


def test_verification_within_timeout_does_nothing():
    m = make_machine(verification_timeout_s=2.0)
    m.on_wake_word(now=0.0)
    actions = m.tick(now=1.0)
    assert m.state == GateState.VERIFYING
    assert actions == []


def test_active_timeout_stops_asr_and_returns_to_idle():
    m = make_machine(active_timeout_s=5.0)
    m.on_wake_word(now=0.0)
    m.on_speaker_verified(now=0.1)
    actions = m.tick(now=10.0)
    assert m.state == GateState.IDLE
    assert GateAction.STOP_ASR in actions
    assert GateAction.RETURN_TO_IDLE in actions


def test_utterance_refreshes_active_window():
    m = make_machine(active_timeout_s=5.0)
    m.on_wake_word(now=0.0)
    m.on_speaker_verified(now=0.1)
    m.on_utterance(now=4.0)  # right before timeout, resets the clock
    actions = m.tick(now=8.0)  # 4s after the utterance, well within 5s budget
    assert m.state == GateState.ACTIVE
    assert actions == []


def test_utterance_ignored_outside_active_state():
    m = make_machine()
    actions = m.on_utterance(now=0.0)
    assert actions == []
    assert m.state == GateState.IDLE


def test_full_happy_path_cycle():
    m = make_machine()
    assert m.on_wake_word(now=0.0) == [GateAction.START_VOICE_ID]
    assert m.state == GateState.VERIFYING
    verified = m.on_speaker_verified(now=0.5)
    assert GateAction.SPEAK_ACCEPT in verified and GateAction.START_ASR in verified
    assert m.state == GateState.ACTIVE
    assert m.on_utterance(now=1.0) == [GateAction.FORWARD_UTTERANCE]
    assert m.state == GateState.ACTIVE


def test_reset_forces_idle_from_any_state():
    m = make_machine()
    m.on_wake_word(now=0.0)
    m.on_speaker_verified(now=0.1)
    assert m.state == GateState.ACTIVE
    m.reset(now=1.0)
    assert m.state == GateState.IDLE
