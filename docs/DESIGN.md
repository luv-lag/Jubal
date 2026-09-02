# Jubal — voice-gated agent

One line: **sherpa-onnx (wake word + ASR + TTS) + a 3D-Speaker/WeSpeaker voice-ID
gate (only your voice wakes it) + Pipecat for the realtime layer + an MCP
server so any agent can use it.**

This is a glue project. Every hard ML problem here (keyword spotting, speech
recognition, speech synthesis, speaker verification) is solved by an
existing, maintained upstream library. Jubal's job is to wire them together
into one coherent gate: *hear the wake word → confirm it's actually you →
transcribe → hand off to an agent → speak the reply.*

## 1. Three-layer architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3 — Interfaces                                                │
│   MCP server (jubal/mcp_server)   Phone/robot clients (stretch)     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2 — Realtime orchestration (Pipecat)                          │
│   AgentGateStateMachine  →  FrameProcessors  →  Pipeline             │
│   (jubal/pipeline)                                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 1 — Model glue (sherpa-onnx + 3D-Speaker/WeSpeaker)            │
│   wake word (KWS) │ voice ID gate │ streaming ASR │ offline ASR │ TTS│
│   (jubal/wake)      (jubal/voiceid)  (jubal/asr)              (jubal/tts) │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer 1 — model glue.** Thin, dependency-injected wrappers around
upstream ONNX runtimes. No pipeline logic lives here, no network calls —
just "give me a wake event", "give me an embedding", "give me text", "give
me audio". Each wrapper matches the actual upstream Python API (see
`docs/MODELS.md` for exact model files and API references).

**Layer 2 — realtime orchestration.** A pure, hardware-free state machine
(`jubal/pipeline/state_machine.py`) owns the gate logic: idle → verifying →
active → back to idle. Pipecat `FrameProcessor`s drive that state machine
off real audio frames and turn its actions into pipeline effects (start
ASR, speak an accept/reject chime, forward an utterance to the agent).
Keeping the state machine free of Pipecat and free of sherpa-onnx is
deliberate: it's the one piece of "our" logic in the whole system, so it's
the one piece worth unit-testing in isolation.

**Layer 3 — interfaces.** An MCP server exposes the same building blocks
(enroll, verify, transcribe, speak) as tools so any MCP-capable agent —
Claude, a robot controller, a home-automation brain — can drive Jubal
without knowing anything about ONNX or audio frames.

## 2. Upstream libraries (don't rebuild any of this)

| Concern | Library | Why |
|---|---|---|
| Wake word / keyword spotting | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) `KeywordSpotter` | Runs fully offline, tiny CPU footprint, ships pretrained KWS models, custom keywords via a keywords file. |
| Streaming ASR | sherpa-onnx `OnlineRecognizer` (zipformer/paraformer transducer) | Same runtime as KWS/TTS — one dependency, one execution provider, incremental `accept_waveform` + endpoint detection. |
| File-based ASR | sherpa-onnx `OfflineRecognizer` | Used by the MCP `transcribe` tool where a whole clip is available up front (better accuracy than forcing the streaming model onto a static file). |
| TTS | sherpa-onnx `OfflineTts` (VITS/Kokoro/Matcha/Kitten backends) | Same runtime family; swap `sid`/model for a different voice without touching glue code. |
| Voice ID / speaker embeddings | **3D-Speaker (ERes2Net / CAM++) or WeSpeaker**, deployed through sherpa-onnx `SpeakerEmbeddingExtractor` | These are the state-of-the-art open speaker-verification embedding families; sherpa-onnx ships ready-to-run ONNX exports, so no training or conversion work is needed. |
| Voice ID fast prototype | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | Pure PyTorch, `pip install`-only, no ONNX export step — good for iterating on gate thresholds/margins before committing to the sherpa-onnx model. Both backends implement the same `EmbeddingExtractor` protocol so the gate logic doesn't care which one produced the vector. |
| Realtime orchestration | [Pipecat](https://github.com/pipecat-ai/pipecat) | Frame-based pipeline abstraction built for exactly this shape of problem (audio in → STT → logic → TTS → audio out); custom `FrameProcessor`s are the documented extension point. |
| Agent interface | [MCP](https://github.com/modelcontextprotocol/python-sdk) (`FastMCP`) | Lets any MCP client (Claude Code, Claude Desktop, a custom agent runtime) call `enroll_voice`, `verify_speaker`, `transcribe`, `speak` as tools instead of talking to a bespoke API. |

## 3. The gate, precisely

```
 mic ──► KWS (wake word) ──► [IDLE → VERIFYING] ──► voice embedding
                                                        │
                                        cosine-similarity vs enrolled
                                        speaker(s), threshold + margin
                                                        │
                              ┌─────────── rejected ────┴──── accepted ───────────┐
                              ▼                                                    ▼
                        speak "not you" / stay silent                    [VERIFYING → ACTIVE]
                        back to IDLE                                     speak accept chime
                                                                          start streaming ASR
                                                                                    │
                                                                          utterance + endpoint
                                                                                    ▼
                                                                          forward text to agent
                                                                                    │
                                                                          agent reply text
                                                                                    ▼
                                                                          TTS ──► speaker out
                                                                                    │
                                                                          inactivity timeout
                                                                          [ACTIVE → IDLE]
```

Verification uses **threshold + margin**, not just threshold: the best
match must clear the similarity threshold *and* beat the second-best
enrolled speaker by a minimum margin, so a mediocre match against you when
someone else is a slightly worse (but still plausible) match doesn't sneak
through. See `jubal/voiceid/gate.py`.

## 4. Honest build plan

Nothing in Layer 1 is novel — it is API glue against upstream examples
(`k2-fsa/sherpa-onnx/python-api-examples/*.py`). The actual engineering
this project contributes is:

1. A backend-agnostic embedding protocol so the voice-ID gate doesn't care
   whether the vector came from sherpa-onnx/3D-Speaker or Resemblyzer.
2. The `AgentGateStateMachine` — the only state a "wake word + voice ID"
   product needs, expressed as plain Python with no I/O, so it can be
   unit-tested without a microphone, a model file, or a GPU.
3. Pipecat `FrameProcessor`s that translate real audio frames into events
   for that state machine, and its actions back into pipeline effects.
4. An MCP server that exposes the same primitives to any agent.

What this project explicitly does **not** do: train or fine-tune any
model, implement speech recognition/synthesis/speaker-embedding math from
scratch, or reimplement Pipecat's frame scheduling.

## 5. MVP scope

1. `scripts/enroll_voice.py` — record/point at a few seconds of your voice,
   extract an embedding (sherpa-onnx or Resemblyzer backend), store it in
   `EmbeddingStore`.
2. Wake word triggers `VERIFYING`; the next ~1–2 s of audio is embedded and
   checked against the enrolled store.
3. On accept: streaming ASR runs until an endpoint is detected.
4. The transcript is POSTed to a configurable agent HTTP endpoint
   (`jubal/pipeline/agent_bridge.py`); the reply text is synthesized with
   sherpa-onnx TTS and played back.
5. All of the above also reachable headless through the MCP server, for
   agents that want to drive Jubal directly instead of through the mic.

## 6. Stretch goals

- Full MCP server with streaming tool calls (partial transcripts as they
  arrive, not just final text).
- Per-user profiles: multiple enrolled speakers, each with their own agent
  routing/permissions — the margin-based gate already returns *which*
  enrolled speaker matched, not just accept/reject.
- Phone client (Pipecat has WebRTC transports; reuse the same pipeline).
- Robot/embedded integration: swap the local `sounddevice` transport for
  whatever audio I/O the robot exposes; the gate and glue layers are
  transport-agnostic.

## 7. Open decisions

- **Pipecat vs LiveKit.** This build uses **Pipecat**, because its
  `FrameProcessor` model maps directly onto "wrap an existing offline
  model as a pipeline stage" with no server infrastructure required, and
  it's usable fully local/offline. LiveKit Agents is the better choice if
  multi-party WebRTC calling or LiveKit's existing telephony/room
  infrastructure becomes a real requirement — swapping is a Layer 2
  change only; Layer 1 (the model glue) is unaffected either way.
- **ASR model size vs latency.** `config/default.yaml` ships pointing at
  small/streaming zipformer models by default (favor latency: wake-to-first-
  partial should be well under a second on CPU). A larger model (or
  Whisper via `OfflineRecognizer.from_whisper`) is a config change, not a
  code change, for setups that can trade latency for accuracy (e.g. the
  file-based MCP `transcribe` tool, where nothing is waiting on a live
  turn).
- **Verification threshold/margin defaults** (`config/default.yaml`,
  `GateConfig`) are a starting point, not tuned numbers — they need
  calibration against your actual enrolled voice and mic before relying on
  this as an access gate for anything sensitive.

See `docs/MODELS.md` for exact model downloads and `README.md` for the
quickstart.
