"""The actual "only your voice wakes it" decision.

Pure, hardware-free, and independent of which embedding backend produced
the vector. Two rules must both pass:

  1. threshold  — the best match must clear a minimum cosine similarity.
  2. margin     — the best match must beat the *second-best* enrolled
                   speaker by a minimum margin, so a mediocre-but-passable
                   match against you doesn't sneak through just because
                   nobody else enrolled is a closer match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jubal.voiceid.store import EmbeddingStore


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class GateConfig:
    threshold: float = 0.72
    margin: float = 0.05


@dataclass
class VerificationResult:
    accepted: bool
    speaker: str | None
    score: float
    runner_up_score: float
    reason: str


class VoiceGate:
    def __init__(self, store: EmbeddingStore, config: GateConfig | None = None):
        self.store = store
        self.config = config or GateConfig()

    def scores(self, embedding: np.ndarray) -> dict[str, float]:
        """Best cosine similarity per enrolled speaker (max over their samples)."""
        result: dict[str, float] = {}
        for name, samples in self.store.all_embeddings().items():
            if samples:
                result[name] = max(cosine_similarity(embedding, s) for s in samples)
        return result

    def evaluate(self, embedding: np.ndarray) -> VerificationResult:
        scores = self.scores(embedding)
        if not scores:
            return VerificationResult(False, None, 0.0, 0.0, "no enrolled speakers")

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score < self.config.threshold:
            return VerificationResult(
                False, None, best_score, runner_up_score,
                f"best match {best_name!r} scored {best_score:.3f}, below threshold "
                f"{self.config.threshold:.3f}",
            )
        if len(ranked) > 1 and (best_score - runner_up_score) < self.config.margin:
            return VerificationResult(
                False, None, best_score, runner_up_score,
                f"ambiguous: {best_name!r} scored {best_score:.3f}, only "
                f"{best_score - runner_up_score:.3f} ahead of runner-up "
                f"(margin required: {self.config.margin:.3f})",
            )
        return VerificationResult(
            True, best_name, best_score, runner_up_score, "accepted",
        )
