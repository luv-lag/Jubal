"""Send a verified, transcribed utterance to the configured agent endpoint
and get back the reply text to speak.

Deliberately a plain POST to a configurable URL rather than a specific
agent framework's SDK: the whole point of the gate is that it hands off to
*any* agent once the user is verified. `jubal/mcp_server` is the other,
agent-initiated direction (agent calls Jubal); this is Jubal calling the
agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from jubal.config import AgentConfig


@dataclass
class AgentReply:
    text: str


class AgentBridge:
    def __init__(self, config: AgentConfig, speaker: str):
        self._config = config
        self._speaker = speaker

    async def send(self, transcript: str) -> AgentReply:
        async with httpx.AsyncClient(timeout=self._config.timeout_s) as client:
            response = await client.post(
                self._config.endpoint_url,
                json={"speaker": self._speaker, "text": transcript},
            )
            response.raise_for_status()
            data = response.json()
        return AgentReply(text=data.get("reply", ""))
