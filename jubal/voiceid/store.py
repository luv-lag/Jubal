"""Enrolled speaker embeddings, persisted as plain JSON.

Deliberately not a sherpa_onnx.SpeakerEmbeddingManager: we want direct
access to every enrolled sample's raw score (for the margin check in
gate.py), and a human-readable, backend-agnostic file that doesn't care
whether an embedding came from sherpa-onnx or Resemblyzer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EmbeddingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, list[list[float]]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        self._data.setdefault(name, []).append(np.asarray(embedding, dtype=np.float32).tolist())

    def remove_speaker(self, name: str) -> bool:
        return self._data.pop(name, None) is not None

    def speakers(self) -> list[str]:
        return list(self._data)

    def embeddings_for(self, name: str) -> list[np.ndarray]:
        return [np.array(e, dtype=np.float32) for e in self._data.get(name, [])]

    def all_embeddings(self) -> dict[str, list[np.ndarray]]:
        return {name: self.embeddings_for(name) for name in self._data}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
