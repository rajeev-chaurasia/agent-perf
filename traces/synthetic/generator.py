"""Synthetic trace generator for agent performance benchmarking."""

from __future__ import annotations

import argparse
import json
import string
import uuid
from pathlib import Path
from typing import Any

import numpy as np

PRESETS_DIR = Path(__file__).parent / "presets"


def _load_preset(preset: str) -> dict[str, Any]:
    preset_path = PRESETS_DIR / f"{preset}.json"
    with preset_path.open() as f:
        return json.load(f)


def _build_shared_prefix(prefix_kb: int) -> str:
    size = prefix_kb * 1024
    chars = string.ascii_letters + string.digits + " \n"
    repeat_unit = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
    )
    repeated = (repeat_unit * ((size // len(repeat_unit)) + 1))[:size]
    return repeated


def generate_trace(
    preset: str,
    sessions: int | None = None,
    turns_mean: float | None = None,
    turns_std: float | None = None,
    prefix_kb: int | None = None,
    tool_call_ratio: float | None = None,
    think_time_ms_mean: int | None = None,
    output_tokens_mean: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    params = _load_preset(preset)

    if sessions is not None:
        params["sessions"] = sessions
    if turns_mean is not None:
        params["turns_mean"] = turns_mean
    if turns_std is not None:
        params["turns_std"] = turns_std
    if prefix_kb is not None:
        params["prefix_kb"] = prefix_kb
    if tool_call_ratio is not None:
        params["tool_call_ratio"] = tool_call_ratio
    if think_time_ms_mean is not None:
        params["think_time_ms_mean"] = think_time_ms_mean
    if output_tokens_mean is not None:
        params["output_tokens_mean"] = output_tokens_mean

    rng = np.random.Generator(np.random.PCG64(seed))

    shared_prefix = _build_shared_prefix(params["prefix_kb"])

    num_sessions = params["sessions"]
    t_mean = params["turns_mean"]
    t_std = params["turns_std"]
    tc_ratio = params["tool_call_ratio"]
    tt_mean = params["think_time_ms_mean"]
    ot_mean = params["output_tokens_mean"]

    sessions_list: list[dict[str, Any]] = []

    for s_idx in range(num_sessions):
        rand_hi = int(rng.integers(0, 2**63))
        rand_lo = int(rng.integers(0, 2**63))
        session_id = f"session-{s_idx:04d}-{uuid.UUID(int=(rand_hi << 63) | rand_lo)}"
        num_turns = max(1, round(rng.normal(t_mean, t_std)))

        turns: list[dict[str, Any]] = []
        for t_idx in range(num_turns):
            output_tokens_hint = int(rng.poisson(ot_mean))
            think_time_ms = int(rng.exponential(tt_mean))
            expects_tool_call = bool(rng.random() < tc_ratio)

            if t_idx == 0:
                role_seq = "user"
            else:
                role_seq = "tool_result" if rng.random() < tc_ratio else "user"

            content_ref = (
                f"content://turns/{session_id}/{t_idx}"
                f"?output_tokens_hint={output_tokens_hint}"
            )

            turn: dict[str, Any] = {
                "turn_id": t_idx,
                "role_sequence": role_seq,
                "content_ref": content_ref,
                "sampling": {
                    "temperature": float(round(rng.uniform(0.0, 1.0), 3)),
                    "max_tokens": int(output_tokens_hint + 64),
                    "top_p": float(round(rng.uniform(0.8, 1.0), 3)),
                },
                "expects_tool_call": expects_tool_call,
                "think_time_ms": think_time_ms,
            }
            turns.append(turn)

        session: dict[str, Any] = {
            "session_id": session_id,
            "system_prompt_ref": shared_prefix,
            "turns": turns,
        }
        sessions_list.append(session)

    trace_id = f"{preset}-seed{seed}"

    return {
        "trace_id": trace_id,
        "schema_version": 1,
        "sessions": sessions_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic agent performance traces."
    )
    parser.add_argument(
        "--preset",
        choices=["chatlike", "agent_shallow", "agent_deep", "agent_swarm"],
        required=True,
        help="Named preset to use as base configuration.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=None,
        help="Override number of sessions.",
    )
    parser.add_argument(
        "--turns-mean",
        type=float,
        default=None,
        dest="turns_mean",
        help="Override mean number of turns per session.",
    )
    parser.add_argument(
        "--turns-std",
        type=float,
        default=None,
        dest="turns_std",
        help="Override std dev of turns per session.",
    )
    parser.add_argument(
        "--prefix-kb",
        type=int,
        choices=[2, 8, 32],
        default=None,
        dest="prefix_kb",
        help="Override shared prefix size in KB.",
    )
    parser.add_argument(
        "--tool-call-ratio",
        type=float,
        default=None,
        dest="tool_call_ratio",
        help="Override tool call ratio (0.0 to 1.0).",
    )
    parser.add_argument(
        "--think-time-ms-mean",
        type=int,
        default=None,
        dest="think_time_ms_mean",
        help="Override mean think time in milliseconds.",
    )
    parser.add_argument(
        "--output-tokens-mean",
        type=int,
        default=None,
        dest="output_tokens_mean",
        help="Override mean output tokens per turn.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to stdout.",
    )

    args = parser.parse_args()

    trace = generate_trace(
        preset=args.preset,
        sessions=args.sessions,
        turns_mean=args.turns_mean,
        turns_std=args.turns_std,
        prefix_kb=args.prefix_kb,
        tool_call_ratio=args.tool_call_ratio,
        think_time_ms_mean=args.think_time_ms_mean,
        output_tokens_mean=args.output_tokens_mean,
        seed=args.seed,
    )

    output_text = json.dumps(trace, indent=2)

    if args.output is not None:
        args.output.write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
