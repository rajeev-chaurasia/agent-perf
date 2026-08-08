"""Replay engine: drives an inference server from a TraceSpec and writes Parquet + manifest."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

from agentperf.clients import MessageTurn, build_client, stream_request
from agentperf.manifest import create_manifest, save_manifest
from agentperf.metrics import e2e_ms, itl_ms, ttft_ms
from agentperf.models import (
    Framework,
    Precision,
    ReplayConfig,
    ReplayMode,
    SessionSpec,
    TraceSpec,
    TurnResult,
)


# --- Content helpers


def resolve_content_ref(ref: str) -> str:
    """If *ref* starts with ``@``, the remainder is a filesystem path whose contents are
    returned verbatim, allowing large prompts to live outside the trace JSON.
    """
    if ref.startswith("@"):
        return Path(ref[1:]).read_text(encoding="utf-8")
    return ref


def build_messages(session: SessionSpec, up_to_turn_id: int) -> list[dict]:
    """TOOL_RESULT turns are surfaced as user messages with a ``[tool_result] `` prefix
    because OpenAI-compatible servers may not support a dedicated tool role.
    """
    messages: list[dict] = [
        {
            "role": "system",
            "content": resolve_content_ref(session.system_prompt_ref),
        }
    ]

    for turn in session.turns:
        if turn.turn_id > up_to_turn_id:
            break
        content = resolve_content_ref(turn.content_ref)
        if turn.role_sequence.value == "tool_result":
            content = f"[tool_result] {content}"
        messages.append({"role": "user", "content": content})

    return messages


# --- Shared session coroutine


async def _replay_session(
    session: SessionSpec,
    client,  # httpx.AsyncClient
    base_url: str,
    run_id: str,
    results: list[TurnResult],
    lock: asyncio.Lock,
    model_name: str = "",
) -> None:
    """Drive a single session sequentially, respecting inter-turn think time."""
    for i, turn in enumerate(session.turns):
        messages = build_messages(session, turn.turn_id)
        mt = MessageTurn(
            messages=messages,
            sampling=turn.sampling,
            session_id=session.session_id,
            turn_id=turn.turn_id,
            run_id=run_id,
            model=model_name,
        )
        result = await stream_request(client, base_url, mt)
        async with lock:
            results.append(result)

        # Apply think time between turns, not after the last one.
        if i < len(session.turns) - 1 and turn.think_time_ms > 0:
            await asyncio.sleep(turn.think_time_ms / 1000)


# --- Replay classes


class ClosedLoopReplay:
    """Replays a trace with a fixed cap on the number of concurrent sessions.

    Uses an asyncio.Semaphore so that at most ``config.concurrency`` sessions
    are active at the same time.  Each session holds the semaphore slot for
    its entire duration (all turns plus inter-turn think time).
    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config

    async def run(
        self, trace: TraceSpec, manifest_kwargs: dict
    ) -> tuple[list[TurnResult], Path]:
        manifest = await create_manifest(
            replay_mode=ReplayMode.CLOSED_LOOP,
            concurrency=self._config.concurrency,
            trace_path=self._config.trace_path,
            **manifest_kwargs,
        )
        run_id = manifest.run_id

        client = build_client(self._config.base_url)
        sem = asyncio.Semaphore(self._config.concurrency)
        all_results: list[TurnResult] = []
        lock = asyncio.Lock()

        async def _bounded_session(session: SessionSpec) -> None:
            # Hold the slot for the full session so concurrency is capped.
            async with sem:
                await _replay_session(
                    session, client, self._config.base_url, run_id, all_results, lock,
                    model_name=self._config.model_name,
                )

        tasks = [asyncio.create_task(_bounded_session(s)) for s in trace.sessions]
        await asyncio.gather(*tasks)
        await client.aclose()

        df = results_to_parquet(all_results, ReplayMode.CLOSED_LOOP, run_id)
        parquet_path = save_parquet(df, self._config.output_dir, run_id)
        await save_manifest(manifest, self._config.output_dir)

        return (all_results, parquet_path)


class OpenLoopReplay:
    """Replays a trace with session arrivals fired at a fixed rate.

    ``config.concurrency`` is interpreted as *arrivals per second*.  Sessions
    are created with ``asyncio.create_task`` so each one starts immediately
    without waiting for the previous to finish.
    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config

    async def run(
        self, trace: TraceSpec, manifest_kwargs: dict
    ) -> tuple[list[TurnResult], Path]:
        manifest = await create_manifest(
            replay_mode=ReplayMode.OPEN_LOOP,
            concurrency=self._config.concurrency,
            trace_path=self._config.trace_path,
            **manifest_kwargs,
        )
        run_id = manifest.run_id

        client = build_client(self._config.base_url)
        all_results: list[TurnResult] = []
        lock = asyncio.Lock()

        # Inter-arrival interval derived from the target arrival rate.
        interval_s: float = 1.0 / self._config.concurrency

        tasks: list[asyncio.Task] = []
        for session in trace.sessions:
            task = asyncio.create_task(
                _replay_session(
                    session, client, self._config.base_url, run_id, all_results, lock,
                    model_name=self._config.model_name,
                )
            )
            tasks.append(task)
            await asyncio.sleep(interval_s)

        await asyncio.gather(*tasks)
        await client.aclose()

        df = results_to_parquet(all_results, ReplayMode.OPEN_LOOP, run_id)
        parquet_path = save_parquet(df, self._config.output_dir, run_id)
        await save_manifest(manifest, self._config.output_dir)

        return (all_results, parquet_path)


# --- Result serialisation


def results_to_parquet(
    results: list[TurnResult], mode: ReplayMode, run_id: str
) -> pd.DataFrame:
    """Latency columns are computed via the canonical metric functions to stay consistent
    with downstream analysis that calls those same functions directly.
    """
    ttft_vals = ttft_ms(results)
    itl_vals = itl_ms(results)
    e2e_vals = e2e_ms(results)

    return pd.DataFrame(
        {
            "session_id": [r.session_id for r in results],
            "turn_id": [r.turn_id for r in results],
            "prompt_tokens": [r.prompt_tokens for r in results],
            "output_tokens": [r.output_tokens for r in results],
            "ttft_ms": ttft_vals,
            "itl_ms": itl_vals,
            "e2e_ms": e2e_vals,
            "http_status": [r.http_status for r in results],
            "cache_hit": [r.cache_hit for r in results],
            "run_id": run_id,
            "replay_mode": mode.value,
        }
    )


def save_parquet(df: pd.DataFrame, output_dir: str, run_id: str) -> Path:
    """Write to ``<output_dir>/<run_id>.parquet``, creating the directory if needed."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dest = out / f"{run_id}.parquet"
    table = pa.Table.from_pandas(df)
    # Embed version metadata so consumers can detect schema compatibility.
    table = table.replace_schema_metadata({b"agentperf_version": b"0.1.0"})
    pq.write_table(table, dest)
    return dest


# --- CLI entry point


def main() -> str:
    parser = argparse.ArgumentParser(
        description="Replay agent traces against a local inference server."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[m.value for m in ReplayMode],
        help="Concurrency model: closed_loop or open_loop.",
    )
    parser.add_argument(
        "--concurrency",
        required=True,
        type=int,
        help="Max concurrent sessions (closed_loop) or arrivals/sec (open_loop).",
    )
    parser.add_argument(
        "--trace",
        required=True,
        type=str,
        help="Path to the TraceSpec JSON file.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the inference server (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for parquet and manifest files (default: results).",
    )
    parser.add_argument(
        "--framework",
        required=True,
        choices=[f.value for f in Framework],
        help="Inference serving framework.",
    )
    parser.add_argument("--model", required=True, type=str, help="Model name or path.")
    parser.add_argument(
        "--precision",
        required=True,
        choices=[p.value for p in Precision],
        help="Model weight precision.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="1",
        help="Comma-separated GPU IDs used for the run (default: 1).",
    )

    args = parser.parse_args()

    console = Console()

    mode = ReplayMode(args.mode)
    framework = Framework(args.framework)
    precision = Precision(args.precision)
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]

    config = ReplayConfig(
        mode=mode,
        concurrency=args.concurrency,
        trace_path=args.trace,
        base_url=args.base_url,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    trace_text = Path(args.trace).read_text(encoding="utf-8")
    trace = TraceSpec.model_validate_json(trace_text)

    manifest_kwargs: dict = {
        "framework": framework,
        "model": args.model,
        "precision": precision,
        "gpu_ids": gpu_ids,
        "clock_mhz_locked": 0,
        "config_dict": vars(args),
    }

    console.print(
        f"[bold]Starting {mode.value} replay[/bold]: "
        f"{len(trace.sessions)} session(s), concurrency={args.concurrency}, "
        f"base_url={args.base_url}"
    )

    replayer: ClosedLoopReplay | OpenLoopReplay
    if mode == ReplayMode.CLOSED_LOOP:
        replayer = ClosedLoopReplay(config)
    else:
        replayer = OpenLoopReplay(config)

    results, parquet_path = asyncio.run(replayer.run(trace, manifest_kwargs))

    console.print(
        f"[green]Done.[/green] {len(results)} turn result(s) written to {parquet_path}"
    )


if __name__ == "__main__":
    main()
