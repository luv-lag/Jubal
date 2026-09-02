# Models

Jubal ships no model weights — it points sherpa-onnx at files you download
once. All of these are published by the sherpa-onnx project (k2-fsa) as
ready-to-run ONNX exports; no conversion step needed.

Model catalogs: https://k2-fsa.github.io/sherpa/onnx/

## Wake word (keyword spotting)

Catalog: https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html

Download a KWS model directory (encoder/decoder/joiner/tokens +
`keywords.txt`), point `config/default.yaml`'s `wake.model_dir` at it, and
edit its `keywords.txt` (or pass a custom keywords string to
`create_stream()`) to set your wake phrase.

## Streaming ASR (realtime pipeline)

Catalog: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html

Default recommendation: a small streaming zipformer transducer (e.g. the
`sherpa-onnx-streaming-zipformer-en-*` small/int8 exports) for low
wake-to-first-partial latency on CPU. `jubal.asr.streaming.StreamingRecognizer`
takes `tokens`/`encoder`/`decoder`/`joiner` paths directly.

## File-based ASR (MCP `transcribe` tool)

Catalog: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
(offline/non-streaming section — transducer, paraformer, or Whisper).

Nothing here is on the clock for a live turn, so a bigger/more accurate
model is a reasonable default; `jubal.asr.offline.OfflineTranscriber`
supports `from_transducer` and `from_whisper` factories.

## Voice ID / speaker embeddings

Catalog: https://k2-fsa.github.io/sherpa/onnx/speaker-identification/pretrained_models/index.html

Recommended: a 3D-Speaker export, e.g.
`3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx` (or the
English/multilingual WeSpeaker exports in the same catalog). Point
`config/default.yaml`'s `voiceid.sherpa_model` at the `.onnx` file.

### Fast prototype path (no ONNX download)

`pip install resemblyzer` and set `voiceid.backend: resemblyzer` in the
config. `jubal.voiceid.backends.resemblyzer_backend.ResemblyzerEmbeddingExtractor`
produces 256-d embeddings from `Resemblyzer.VoiceEncoder` — good enough to
tune gate thresholds/margins before switching to the sherpa-onnx backend
for production. Both backends satisfy the same `EmbeddingExtractor`
protocol, so nothing else in the codebase changes when you switch.

## TTS

Voice samples/catalog: https://k2-fsa.github.io/sherpa/onnx/tts/all/index.html

Default recommendation: a `vits-piper-en_US-*` voice (small, fast,
good-enough quality for a spoken reply). `jubal.tts.synthesizer.Synthesizer`
only wires up the VITS backend today — sherpa-onnx also exposes
Kokoro/Matcha/Kitten offline TTS models through the same `OfflineTtsConfig`
shape, so adding one is a small, isolated change if you want a
higher-quality voice and can afford the extra latency.

A `model_dir` for a `vits-piper-*` voice should contain `model.onnx`,
`tokens.txt`, and (for piper voices) an `espeak-ng-data/` directory —
that's what `Synthesizer` expects.

## Where paths go

All of the above are referenced from `config/default.yaml` — copy it,
fill in the paths for whatever you downloaded, and point
`scripts/run_agent.py` / `jubal.mcp_server.server` at your copy via
`JUBAL_CONFIG`.
