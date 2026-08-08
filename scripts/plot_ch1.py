"""Generate Chapter 1 result plots from collected Parquet files."""
from __future__ import annotations

import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

RESULTS_DIR = pathlib.Path("chapters/ch1_frameworks/results")
OUT_DIR = pathlib.Path("docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FW_COLORS = {
    "vllm":   "#2563EB",
    "sglang": "#16A34A",
    "trtllm": "#DC2626",
}
FW_LABELS = {
    "vllm":   "vLLM 0.26",
    "sglang": "SGLang 0.5.9",
    "trtllm": "TRT-LLM 1.3",
}
TRACE_LABELS = {
    "agent_shallow": "Shallow (20 sessions, ~23 short turns)",
    "agent_deep":    "Deep (20 sessions, ~23 growing-context turns)",
    "agent_swarm":   "Swarm (100 sessions, 3 short turns each)",
}
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128]
FRAMEWORKS  = ["vllm", "sglang", "trtllm"]
TRACES      = ["agent_shallow", "agent_deep", "agent_swarm"]


def load_all() -> pd.DataFrame:
    path_re = re.compile(
        r".*/(?P<fw>vllm|sglang|trtllm)/(?P<trace>[^/]+)/run(?P<run>\d+)/c(?P<c>\d+)/.*\.parquet$"
    )
    frames = []
    for f in RESULTS_DIR.rglob("*.parquet"):
        if "warmup" in str(f):
            continue
        m = path_re.match(str(f))
        if not m:
            continue
        df = pd.read_parquet(f)
        df["framework"]  = m.group("fw")
        df["trace"]      = m.group("trace")
        df["run"]        = int(m.group("run"))
        df["concurrency"] = int(m.group("c"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def p50(series: pd.Series) -> float:
    valid = series[(series != -1.0) & series.notna()]
    return float(valid.quantile(0.50)) if len(valid) else float("nan")


def p99(series: pd.Series) -> float:
    valid = series[(series != -1.0) & series.notna()]
    return float(valid.quantile(0.99)) if len(valid) else float("nan")


def agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return p50 per (framework, trace, concurrency), averaged across runs."""
    rows = []
    for (fw, trace, c), g in df.groupby(["framework", "trace", "concurrency"]):
        rows.append({"framework": fw, "trace": trace, "concurrency": c,
                     "p50": p50(g[metric]), "p99": p99(g[metric])})
    return pd.DataFrame(rows).sort_values("concurrency")


# ── Plot 1: TTFT p50 vs concurrency (3×1 subplots, one per trace) ──────────

def plot_ttft_vs_concurrency(df: pd.DataFrame) -> None:
    agg_df = agg(df, "ttft_ms")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Time to First Token (p50) vs Concurrency", fontsize=15, fontweight="bold", y=1.02)

    for ax, trace in zip(axes, TRACES):
        sub = agg_df[agg_df["trace"] == trace]
        for fw in FRAMEWORKS:
            row = sub[sub["framework"] == fw].sort_values("concurrency")
            if row.empty:
                continue
            ax.plot(row["concurrency"], row["p50"],
                    marker="o", linewidth=2, markersize=6,
                    color=FW_COLORS[fw], label=FW_LABELS[fw])

        ax.set_title(TRACE_LABELS[trace], fontsize=10, pad=8)
        ax.set_xlabel("Concurrent sessions", fontsize=10)
        ax.set_ylabel("TTFT p50 (ms)" if ax == axes[0] else "", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONCURRENCY)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "ttft_vs_concurrency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote ttft_vs_concurrency.png")


# ── Plot 2: E2E p50 at c=128 per framework and trace ───────────────────────

def plot_e2e_high_concurrency(df: pd.DataFrame) -> None:
    high = df[df["concurrency"] == 128]
    agg_df = agg(high, "e2e_ms")

    x = np.arange(len(TRACES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, fw in enumerate(FRAMEWORKS):
        vals = []
        for trace in TRACES:
            sub = agg_df[(agg_df["framework"] == fw) & (agg_df["trace"] == trace)]
            vals.append(sub["p50"].values[0] if len(sub) else float("nan"))
        bars = ax.bar(x + i * width, vals, width, label=FW_LABELS[fw],
                      color=FW_COLORS[fw], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("End-to-End Latency p50 at Concurrency = 128", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("E2E p50 (ms)", fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Shallow", "Deep", "Swarm"], fontsize=11)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "e2e_high_concurrency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote e2e_high_concurrency.png")


# ── Plot 3: TTFT heatmap — p50 across framework × concurrency for each trace

def plot_ttft_heatmap(df: pd.DataFrame) -> None:
    agg_df = agg(df, "ttft_ms")
    selected_c = [1, 4, 16, 32, 64, 128]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("TTFT p50 (ms) — Framework vs Concurrency", fontsize=14, fontweight="bold", y=1.03)

    for ax, trace in zip(axes, TRACES):
        matrix = []
        for fw in FRAMEWORKS:
            row_vals = []
            for c in selected_c:
                sub = agg_df[(agg_df["framework"] == fw) &
                             (agg_df["trace"] == trace) &
                             (agg_df["concurrency"] == c)]
                row_vals.append(sub["p50"].values[0] if len(sub) else float("nan"))
            matrix.append(row_vals)

        mat = np.array(matrix)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=np.nanmin(mat) * 0.9, vmax=np.nanmax(mat) * 1.05)

        ax.set_xticks(range(len(selected_c)))
        ax.set_xticklabels([str(c) for c in selected_c], fontsize=9)
        ax.set_yticks(range(len(FRAMEWORKS)))
        ax.set_yticklabels([FW_LABELS[fw] for fw in FRAMEWORKS], fontsize=9)
        ax.set_xlabel("Concurrent sessions", fontsize=9)
        ax.set_title(TRACE_LABELS[trace].split(" (")[0], fontsize=10, fontweight="bold")

        for i in range(len(FRAMEWORKS)):
            for j in range(len(selected_c)):
                val = mat[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > np.nanmedian(mat) * 1.2 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=text_color)

        plt.colorbar(im, ax=ax, shrink=0.85, label="ms")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "ttft_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote ttft_heatmap.png")


# ── Plot 4: Goodput (tokens/s meeting SLOs) vs concurrency ─────────────────

def plot_goodput(df: pd.DataFrame) -> None:
    TTFT_SLO = 1000.0
    ITL_SLO  = 50.0

    rows = []
    for (fw, trace, run, c), g in df.groupby(["framework", "trace", "run", "concurrency"]):
        ttft_ok = (g["ttft_ms"] > 0) & (g["ttft_ms"] < TTFT_SLO)
        itl_ok  = (g["itl_ms"] < 0) | (g["itl_ms"] < ITL_SLO)
        compliant = g[ttft_ok & itl_ok]
        window_s  = g["e2e_ms"].max() / 1000.0 if len(g) else 0
        gput = (compliant["output_tokens"].sum() / window_s) if window_s > 0 else 0
        rows.append({"framework": fw, "trace": trace, "run": run, "concurrency": c, "goodput": gput})

    gdf = pd.DataFrame(rows)
    # Average across runs
    gmean = gdf.groupby(["framework", "trace", "concurrency"])["goodput"].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(
        f"Goodput (output tokens/s, TTFT < {TTFT_SLO:.0f} ms & ITL < {ITL_SLO:.0f} ms)",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, trace in zip(axes, TRACES):
        sub = gmean[gmean["trace"] == trace]
        for fw in FRAMEWORKS:
            row = sub[sub["framework"] == fw].sort_values("concurrency")
            if row.empty:
                continue
            ax.plot(row["concurrency"], row["goodput"],
                    marker="s", linewidth=2, markersize=6,
                    color=FW_COLORS[fw], label=FW_LABELS[fw])

        ax.set_title(TRACE_LABELS[trace].split(" (")[0], fontsize=10, pad=8)
        ax.set_xlabel("Concurrent sessions", fontsize=10)
        ax.set_ylabel("Goodput (tokens/s)" if ax == axes[0] else "", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONCURRENCY)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "goodput_vs_concurrency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote goodput_vs_concurrency.png")


if __name__ == "__main__":
    print("Loading all parquet files...")
    df = load_all()
    print(f"Loaded {len(df):,} rows from {df['framework'].nunique()} frameworks, "
          f"{df['trace'].nunique()} traces, {df['concurrency'].nunique()} concurrency levels")

    plot_ttft_vs_concurrency(df)
    plot_e2e_high_concurrency(df)
    plot_ttft_heatmap(df)
    plot_goodput(df)
    print("All plots written to docs/images/")
