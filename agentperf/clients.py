"""HTTP client for single-turn streaming requests to an OpenAI-compatible server."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from agentperf.models import SamplingConfig, TurnResult


# --- MessageTurn

@dataclass
class MessageTurn:
    messages: list[dict]
    sampling: SamplingConfig
    session_id: str
    turn_id: int
    run_id: str
    model: str = ""


# --- Client factory

def build_client(base_url: str, timeout_s: float = 300.0) -> httpx.AsyncClient:  # noqa: ARG001
    """``base_url`` is accepted for interface symmetry but is NOT bound to the returned
    client -- pass it explicitly to ``stream_request`` so one client can fan out to
    multiple servers.
    """
    limits = httpx.Limits(
        max_connections=256,
        max_keepalive_connections=256,
    )
    timeout = httpx.Timeout(timeout_s)
    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        http2=False,
    )


# --- Core request function

async def stream_request(
    client: httpx.AsyncClient,
    base_url: str,
    turn: MessageTurn,
) -> TurnResult:
    """Never raises. All error conditions produce a TurnResult with zero/sentinel values."""
    url = base_url.rstrip("/") + "/v1/chat/completions"

    payload: dict = {
        "model": turn.model,
        "messages": turn.messages,
        "temperature": turn.sampling.temperature,
        "max_tokens": turn.sampling.max_tokens,
        "top_p": turn.sampling.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    first_token_ns: int = 0
    last_token_ns: int = 0
    prompt_tokens: int = -1
    output_tokens: int = -1

    request_sent_ns: int = time.monotonic_ns()

    try:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                return TurnResult(
                    session_id=turn.session_id,
                    turn_id=turn.turn_id,
                    run_id=turn.run_id,
                    request_sent_ns=request_sent_ns,
                    first_token_ns=0,
                    last_token_ns=0,
                    prompt_tokens=-1,
                    output_tokens=-1,
                    http_status=response.status_code,
                )

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()

                if data_str == "[DONE]":
                    last_token_ns = time.monotonic_ns()
                    continue

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if first_token_ns == 0:
                    choices = chunk.get("choices") or []
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:  # non-empty string
                            first_token_ns = time.monotonic_ns()
                            break

                # Extract usage (present in the final chunk when include_usage=True)
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", -1)
                    output_tokens = usage.get("completion_tokens", -1)

    except httpx.HTTPError:
        return TurnResult(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            run_id=turn.run_id,
            request_sent_ns=request_sent_ns,
            first_token_ns=0,
            last_token_ns=0,
            prompt_tokens=-1,
            output_tokens=-1,
            http_status=0,
        )

    return TurnResult(
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        run_id=turn.run_id,
        request_sent_ns=request_sent_ns,
        first_token_ns=first_token_ns,
        last_token_ns=last_token_ns,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        http_status=200,
    )
