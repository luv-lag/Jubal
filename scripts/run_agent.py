#!/usr/bin/env python3
"""Run the full realtime pipeline: wake word -> voice-ID gate -> ASR -> agent -> TTS.

    python scripts/run_agent.py --config config/default.yaml

Thin wrapper around jubal.pipeline.run_pipecat; kept as a separate entry
point so `python -m jubal.pipeline.run_pipecat` and a plain `scripts/`
invocation both work.
"""

from jubal.pipeline.run_pipecat import main

if __name__ == "__main__":
    main()
