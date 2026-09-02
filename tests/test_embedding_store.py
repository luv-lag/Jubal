import numpy as np

from jubal.voiceid.store import EmbeddingStore


def test_enroll_and_retrieve(tmp_path):
    store = EmbeddingStore(tmp_path / "store.json")
    store.enroll("jordan", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    store.enroll("jordan", np.array([0.9, 0.1, 0.0], dtype=np.float32))

    assert store.speakers() == ["jordan"]
    assert len(store.embeddings_for("jordan")) == 2


def test_save_and_reload_round_trips(tmp_path):
    path = tmp_path / "store.json"
    store = EmbeddingStore(path)
    store.enroll("jordan", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    store.save()

    reloaded = EmbeddingStore(path)
    assert reloaded.speakers() == ["jordan"]
    np.testing.assert_allclose(reloaded.embeddings_for("jordan")[0], [1.0, 2.0, 3.0])


def test_missing_file_starts_empty(tmp_path):
    store = EmbeddingStore(tmp_path / "does-not-exist.json")
    assert store.speakers() == []
    assert store.embeddings_for("nobody") == []


def test_remove_speaker(tmp_path):
    store = EmbeddingStore(tmp_path / "store.json")
    store.enroll("jordan", np.array([1.0, 0.0], dtype=np.float32))
    assert store.remove_speaker("jordan") is True
    assert store.speakers() == []
    assert store.remove_speaker("jordan") is False
