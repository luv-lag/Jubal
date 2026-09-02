import numpy as np
import pytest

from jubal.voiceid.gate import GateConfig, VoiceGate, cosine_similarity
from jubal.voiceid.store import EmbeddingStore


def unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_cosine_similarity_identical_vectors_is_one():
    v = unit([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_defined_as_zero():
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_no_enrolled_speakers_is_rejected(tmp_path):
    gate = VoiceGate(EmbeddingStore(tmp_path / "s.json"), GateConfig())
    result = gate.evaluate(unit([1.0, 0.0, 0.0]))
    assert not result.accepted
    assert "no enrolled speakers" in result.reason


def test_close_match_above_threshold_is_accepted(tmp_path):
    store = EmbeddingStore(tmp_path / "s.json")
    store.enroll("jordan", unit([1.0, 0.0, 0.0]))
    gate = VoiceGate(store, GateConfig(threshold=0.9, margin=0.05))

    result = gate.evaluate(unit([0.99, 0.01, 0.0]))
    assert result.accepted
    assert result.speaker == "jordan"
    assert result.score > 0.9


def test_far_match_below_threshold_is_rejected(tmp_path):
    store = EmbeddingStore(tmp_path / "s.json")
    store.enroll("jordan", unit([1.0, 0.0, 0.0]))
    gate = VoiceGate(store, GateConfig(threshold=0.9, margin=0.05))

    result = gate.evaluate(unit([0.0, 1.0, 0.0]))
    assert not result.accepted
    assert result.speaker is None
    assert "below threshold" in result.reason


def test_ambiguous_match_within_margin_is_rejected(tmp_path):
    store = EmbeddingStore(tmp_path / "s.json")
    store.enroll("jordan", unit([1.0, 0.0, 0.0]))
    store.enroll("alex", unit([0.99, 0.02, 0.0]))
    gate = VoiceGate(store, GateConfig(threshold=0.5, margin=0.05))

    # nearly equidistant from both enrolled speakers
    result = gate.evaluate(unit([0.995, 0.01, 0.0]))
    assert not result.accepted
    assert "ambiguous" in result.reason


def test_best_of_multiple_samples_per_speaker_is_used(tmp_path):
    store = EmbeddingStore(tmp_path / "s.json")
    store.enroll("jordan", unit([0.0, 1.0, 0.0]))  # a bad sample
    store.enroll("jordan", unit([1.0, 0.0, 0.0]))  # a great sample
    gate = VoiceGate(store, GateConfig(threshold=0.9, margin=0.05))

    result = gate.evaluate(unit([1.0, 0.0, 0.0]))
    assert result.accepted
    assert result.speaker == "jordan"
