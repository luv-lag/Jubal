#!/usr/bin/env python3
"""Enroll a speaker from one or more wav clips.

    python scripts/enroll_voice.py --config config/default.yaml \
        --name jordan sample1.wav sample2.wav sample3.wav

A few short (3-10s), clean clips beat one long noisy one — each clip is
enrolled as a separate sample and the gate matches against the best of them.
"""

from __future__ import annotations

import argparse

from jubal.audio_io import read_wave
from jubal.config import JubalConfig
from jubal.voiceid.factory import build_extractor
from jubal.voiceid.store import EmbeddingStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JubalConfig YAML file")
    parser.add_argument("--name", required=True, help="Speaker name to enroll")
    parser.add_argument("wav_files", nargs="+", help="One or more wav clips of this speaker")
    args = parser.parse_args()

    config = JubalConfig.load(args.config)
    extractor = build_extractor(config.voiceid)
    store = EmbeddingStore(config.voiceid.store_path)

    for wav_path in args.wav_files:
        samples, sample_rate = read_wave(wav_path)
        embedding = extractor.extract(samples, sample_rate)
        store.enroll(args.name, embedding)
        print(f"enrolled sample from {wav_path}")

    store.save()
    print(f"{args.name!r} now has {len(store.embeddings_for(args.name))} sample(s) in {store.path}")


if __name__ == "__main__":
    main()
