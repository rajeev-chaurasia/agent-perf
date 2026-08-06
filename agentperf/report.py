"""Reporting module: load Parquet results, produce plots and markdown tables.

All plots are saved as PNG files using a non-interactive backend so this
module is safe to import in headless / CI environments.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Columns every result Parquet file must contain.
REQUIRED_COLUMNS: List[str] = [
    "session_id",
    "turn_id",
    "prompt_tokens",
    "output_tokens",
    "ttft_ms",
    "itl_ms",
    "e2e_ms",
    "http_status",
    "cache_hit",
    "run_id",
    "replay_mode",
]

# Type alias used throughout: (label, dataframe) pairs.
LabeledRun = Tuple[str, pd.DataFrame]


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_run(parquet_path: str) -> pd.DataFrame:
    """Load a single result Parquet file and validate its schema.

    Raises ValueError when required columns are absent so callers get an
    actionable message rather than a cryptic KeyError later.
    """
    df = pd.read_parquet(parquet_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet file '{parquet_path}' is missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )
    return df


# ── Validation helpers ─────────────────────────────────────────────────────────

def _check_mode_consistency(labeled_runs: List[LabeledRun]) -> None:
    """Verify that all runs share the same replay_mode.

    Mixing closed-loop and open-loop measurements in a single chart produces
    misleading comparisons because their concurrency semantics differ.
    """
    modes = {
        label: df["replay_mode"].iloc[0]
        for label, df in labeled_runs
        if not df.empty
    }
    unique_modes = set(modes.values())
    if len(unique_modes) > 1:
        detail = ", ".join(f"'{lbl}' → {mode}" for lbl, mode in modes.items())
        raise ValueError(
            f"Cannot mix replay modes in a single chart ({detail}). "
            "Run separate reports per mode."
        )


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_goodput_vs_concurrency(
    labeled_runs: List[LabeledRun],
    output_path: str,
    slo_ttft_ms: float = 1000.0,
    slo_itl_ms: float = 50.0,
) -> None:
    """Bar/line chart of SLO-compliant token throughput for each labeled run.

    Goodput counts only output tokens from requests that satisfied both the
    TTFT SLO and the ITL SLO (itl_ms < 0 means streaming latency was not
    measured for that row — those rows pass the ITL check automatically).

    x-axis: run label (one point per concurrency level / configuration).
    y-axis: goodput in tokens/second.
    """
    _check_mode_consistency(labeled_runs)

    replay_mode = labeled_runs[0][1]["replay_mode"].iloc[0] if labeled_runs else "unknown"

    labels: List[str] = []
    goodputs: List[float] = []

    for label, df in labeled_runs:
        ttft_ok = (df["ttft_ms"] > 0) & (df["ttft_ms"] < slo_ttft_ms)
        # Rows where itl_ms was not captured (< 0) are exempt from the ITL SLO.
        itl_ok = (df["itl_ms"] < 0) | (df["itl_ms"] < slo_itl_ms)
        compliant = df[ttft_ok & itl_ok]

        total_tokens = compliant["output_tokens"].sum()
        # Wall-clock window in seconds; guard against zero division.
        window_s = df["e2e_ms"].max() / 1000.0 if not df.empty else 0.0
        goodput = total_tokens / window_s if window_s > 0 else 0.0

        labels.append(label)
        goodputs.append(goodput)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    ax.bar(labels, goodputs, color="steelblue", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Run label")
    ax.set_ylabel("Goodput (output tokens / s)")
    ax.set_title(
        f"Goodput vs. Concurrency  |  mode={replay_mode}  "
        f"|  SLO: TTFT<{slo_ttft_ms}ms, ITL<{slo_itl_ms}ms"
    )
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cache_hit_vs_depth(
    labeled_runs: List[LabeledRun],
    output_path: str,
) -> None:
    """Line chart showing how prefix-cache hit rate evolves with turn depth.

    Turn depth (turn_id) serves as a proxy for context length.  A rising
    hit rate indicates the KV cache is being reused across sessions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, df in labeled_runs:
        # cache_hit may be stored as bool or 0/1 int; cast to float for mean().
        grouped = (
            df.groupby("turn_id")["cache_hit"]
            .apply(lambda s: s.astype(float).mean())
            .reset_index()
        )
        ax.plot(grouped["turn_id"], grouped["cache_hit"], marker="o", label=label)

    ax.set_xlabel("Turn depth (turn_id)")
    ax.set_ylabel("Mean cache hit rate")
    ax.set_title("Cache Hit Rate vs. Turn Depth")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ttft_cdf(
    labeled_runs: List[LabeledRun],
    output_path: str,
) -> None:
    """Empirical CDF of time-to-first-token latency for each labeled run.

    Rows where ttft_ms == -1.0 (errors or non-streaming requests) are
    excluded because they do not represent measured latencies.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, df in labeled_runs:
        valid = df.loc[df["ttft_ms"] != -1.0, "ttft_ms"].sort_values()
        if valid.empty:
            continue
        fraction = [i / len(valid) for i in range(1, len(valid) + 1)]
        ax.plot(valid.values, fraction, label=label)

    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("TTFT Empirical CDF")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_precision_ladder(
    labeled_runs: List[LabeledRun],
    output_path: str,
) -> None:
    """Bar chart comparing median token throughput across precision configurations.

    Only HTTP 200 rows are considered; error responses inflate e2e_ms and
    produce zero output tokens, which would unfairly depress median figures.
    Throughput is output_tokens / (e2e_ms / 1000) per row.
    """
    labels: List[str] = []
    median_throughputs: List[float] = []

    for label, df in labeled_runs:
        ok = df[df["http_status"] == 200].copy()
        # Avoid division by zero for rows that somehow have e2e_ms == 0.
        ok = ok[ok["e2e_ms"] > 0]
        if ok.empty:
            throughput = 0.0
        else:
            per_row = ok["output_tokens"] / (ok["e2e_ms"] / 1000.0)
            throughput = float(per_row.median())

        labels.append(label)
        median_throughputs.append(throughput)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    ax.bar(labels, median_throughputs, color="darkorange", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Configuration (label)")
    ax.set_ylabel("Median throughput (output tokens / s)")
    ax.set_title("Precision Ladder — Median Token Throughput")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Markdown table ─────────────────────────────────────────────────────────────

def markdown_table(
    labeled_runs: List[LabeledRun],
    metrics: List[str],
) -> str:
    """Produce a GitHub-Flavored Markdown table with median and p99 per metric.

    Sentinel values (-1.0) are excluded before computing statistics so that
    missing measurements do not distort aggregate percentiles.
    """
    # Build header: Run | metric_median | metric_p99 | …
    header_cols = ["Run"]
    for m in metrics:
        header_cols += [f"{m}_median", f"{m}_p99"]

    header = "| " + " | ".join(header_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(header_cols)) + " |"

    rows: List[str] = [header, separator]

    for label, df in labeled_runs:
        row_values: List[str] = [label]
        for m in metrics:
            if m not in df.columns:
                row_values += ["N/A", "N/A"]
                continue
            # Strip sentinel -1.0 values before aggregation.
            series = df.loc[df[m] != -1.0, m].dropna()
            if series.empty:
                row_values += ["N/A", "N/A"]
            else:
                median_val = series.median()
                p99_val = series.quantile(0.99)
                row_values += [f"{median_val:.2f}", f"{p99_val:.2f}"]
        rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join(rows)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """Command-line interface for generating report plots and tables."""
    parser = argparse.ArgumentParser(
        description="Generate performance report plots and tables from Parquet result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        metavar="PATH",
        nargs="+",
        required=True,
        help="One or more Parquet result files to load.",
    )
    parser.add_argument(
        "--labels",
        metavar="STR",
        nargs="+",
        required=True,
        help="Human-readable label for each Parquet file (must match --parquet count).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="report_output",
        help="Directory where PNG plots and table output are written.",
    )
    parser.add_argument(
        "--plot",
        choices=["goodput", "cache", "ttft", "precision", "all"],
        default="all",
        help="Which plot(s) to generate.",
    )
    parser.add_argument(
        "--slo-ttft-ms",
        type=float,
        default=1000.0,
        help="TTFT SLO threshold in milliseconds for goodput calculation.",
    )
    parser.add_argument(
        "--slo-itl-ms",
        type=float,
        default=50.0,
        help="ITL SLO threshold in milliseconds for goodput calculation.",
    )

    args = parser.parse_args()

    if len(args.parquet) != len(args.labels):
        parser.error(
            f"--parquet and --labels must have the same number of arguments "
            f"(got {len(args.parquet)} paths and {len(args.labels)} labels)."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_runs: List[LabeledRun] = [
        (label, load_run(path))
        for label, path in zip(args.labels, args.parquet)
    ]

    want_all = args.plot == "all"

    if want_all or args.plot == "goodput":
        plot_goodput_vs_concurrency(
            labeled_runs,
            output_path=str(out_dir / "goodput_vs_concurrency.png"),
            slo_ttft_ms=args.slo_ttft_ms,
            slo_itl_ms=args.slo_itl_ms,
        )

    if want_all or args.plot == "cache":
        plot_cache_hit_vs_depth(
            labeled_runs,
            output_path=str(out_dir / "cache_hit_vs_depth.png"),
        )

    if want_all or args.plot == "ttft":
        plot_ttft_cdf(
            labeled_runs,
            output_path=str(out_dir / "ttft_cdf.png"),
        )

    if want_all or args.plot == "precision":
        plot_precision_ladder(
            labeled_runs,
            output_path=str(out_dir / "precision_ladder.png"),
        )

    # Always emit a summary markdown table for the primary latency metrics.
    table = markdown_table(labeled_runs, metrics=["ttft_ms", "itl_ms", "e2e_ms"])
    table_path = out_dir / "summary_table.md"
    table_path.write_text(table)
    print(table)


if __name__ == "__main__":
    main()
