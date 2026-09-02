# Jubal

Voice-gated agent: **sherpa-onnx (wake word + ASR + TTS) + a 3D-Speaker/
WeSpeaker voice-ID gate (only your voice wakes it) + Pipecat for the
realtime layer + an MCP server so any agent can use it.**

Glue, not from scratch — see `docs/DESIGN.md` for the full architecture,
the honest build plan, and the open decisions (Pipecat vs LiveKit, ASR
model size vs latency). See `docs/MODELS.md` for exactly which model
files to download and where they go.

## Layout

```
jubal/
  config.py              typed config, loaded from YAML
  audio_io.py             wav read/write helpers
  wake/                   keyword spotting (sherpa-onnx KeywordSpotter)
  voiceid/                the gate: embedding store, threshold+margin
                           decision logic, sherpa-onnx & Resemblyzer backends
  asr/                    streaming + offline transcription (sherpa-onnx)
  tts/                    speech synthesis (sherpa-onnx)
  pipeline/               AgentGateStateMachine (pure, testable) +
                           Pipecat FrameProcessors that drive it + the
                           agent HTTP bridge
  mcp_server/             MCP server exposing enroll/verify/transcribe/speak
scripts/
  enroll_voice.py          CLI to enroll a speaker from wav clips
  run_agent.py             run the full realtime pipeline
config/default.yaml        copy this, fill in your model paths
tests/                     unit tests for the parts that don't need a
                           microphone, model files, or a GPU
```

## Install

```bash
pip install -e .                 # core: config, voice-ID gate/store logic
pip install -e ".[sherpa]"       # + sherpa-onnx (wake word, ASR, TTS, voiceid backend)
pip install -e ".[prototype]"    # + Resemblyzer (fast voice-ID prototyping, no ONNX export)
pip install -e ".[realtime]"     # + Pipecat (the live mic/speaker pipeline)
pip install -e ".[mcp]"          # + the MCP SDK (the agent-facing server)
pip install -e ".[dev]"          # + pytest
```

## Quickstart (MVP)

1. Download models per `docs/MODELS.md` and copy `config/default.yaml`,
   filling in the paths.
2. Enroll your voice:
   ```bash
   python scripts/enroll_voice.py --config config/my.yaml --name you sample1.wav sample2.wav
   ```
3. Point `agent.endpoint_url` in your config at an HTTP endpoint that
   accepts `{"speaker": str, "text": str}` and returns `{"reply": str}`.
4. Run the realtime pipeline:
   ```bash
   python scripts/run_agent.py --config config/my.yaml
   ```
   Say the wake word, then speak — if the voice-ID gate matches you above
   `voiceid.threshold` (with a large enough margin over anyone else
   enrolled), it transcribes, forwards to your agent, and speaks the reply.

Or skip the mic entirely and drive Jubal from any MCP client:

```bash
JUBAL_CONFIG=config/my.yaml python -m jubal.mcp_server.server
```

exposes `enroll_voice`, `list_enrolled_speakers`, `verify_speaker`,
`transcribe`, and `speak` as MCP tools.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite covers the config loader, the embedding store, and —most
importantly— the voice-ID gate's threshold/margin decision and the
`AgentGateStateMachine`'s state transitions and timeouts, all without
needing sherpa-onnx, Pipecat, a microphone, or model files installed.
Everything that talks to an actual ONNX model, Pipecat pipeline, or MCP
transport is verified against the real upstream APIs (see `docs/DESIGN.md`
§4) but isn't exercised end-to-end here — that needs real audio hardware
and downloaded model weights, which this environment doesn't have.
