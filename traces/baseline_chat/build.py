# Dataset: ShareGPT https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def count_human_turns(conversation: list[dict]) -> int:
    return sum(1 for msg in conversation if msg.get("from") == "human")


def convert_conversation(conversation: list[dict]) -> dict:
    """Convert a ShareGPT conversation to a trace session."""
    system_prompt_ref: Optional[str] = None
    turns = []

    for msg in conversation:
        speaker = msg.get("from", "")
        value = msg.get("value", "")

        if speaker == "system":
            if system_prompt_ref is None:
                system_prompt_ref = value
        elif speaker == "human":
            turn_spec = {
                "role_sequence": "user",
                "content": value,
                "expects_tool_call": False,
                "think_time_ms": 300,
                "sampling": {
                    "temperature": 0.7,
                    "max_tokens": 512,
                },
            }
            turns.append(turn_spec)
        elif speaker == "gpt":
            # gpt messages are skipped (not replayed)
            pass

    if system_prompt_ref is None:
        system_prompt_ref = DEFAULT_SYSTEM_PROMPT

    return {
        "system_prompt_ref": system_prompt_ref,
        "turns": turns,
    }


def convert(
    input_path: str, output_path: str, max_sessions: int, seed: int
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: input must be a JSON array of ShareGPT conversation objects.", file=sys.stderr)
        sys.exit(1)

    # Filter: keep conversations with 1-4 human turns
    filtered = [
        item
        for item in data
        if isinstance(item, dict)
        and "conversations" in item
        and 1 <= count_human_turns(item["conversations"]) <= 4
    ]

    if len(filtered) > max_sessions:
        rng = random.Random(seed)
        filtered = rng.sample(filtered, max_sessions)

    sessions = []
    for item in filtered:
        session = convert_conversation(item["conversations"])
        sessions.append(session)

    output = {
        "trace_id": "baseline-chat-v1",
        "schema_version": 1,
        "sessions": sessions,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(sessions)} sessions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT conversations to agent-perf baseline chat trace JSON."
    )
    parser.add_argument("--input", required=True, help="Path to ShareGPT JSON file")
    parser.add_argument("--output", required=True, help="Path to write output trace JSON")
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=500,
        dest="max_sessions",
        help="Maximum number of sessions to include (default: 500)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    convert(args.input, args.output, args.max_sessions, args.seed)


if __name__ == "__main__":
    main()
