# agent-perf

Chat benchmarks miss the point for agentic deployments. Agent traffic has three distinctive
properties: (1) deep multi-turn sessions with large shared system prompts create sustained
prefix KV reuse pressure, (2) tool-call turns are bursty with short outputs that stress the
scheduler differently than chat, (3) many concurrent agent sessions race for KV cache capacity.
This suite quantifies what these differences mean for TTFT, ITL, throughput, and goodput across
vLLM, SGLang, and TensorRT-LLM on 96 GB workstation GPU (sm_120, 96 GB) — the only public
benchmark data on this hardware for agentic workloads.

---

## Hardware

| | |
|---|---|
| GPUs | 2x 96 GB workstation GPU, 96 GB each (sm_120, 188 SMs) |
| Interconnect | PCIe Gen5 (no NVLink) |
| CUDA | 13.2 |
| GPU assignment | GPU 0: display; GPU 1: primary benchmark GPU; both GPUs used for TP=2 chapters |

---

## Quick Start (no GPU needed for smoke test)

```bash
pip install -e ".[dev]"

python traces/synthetic/generator.py \
    --preset agent_shallow \
    --output smoke_trace.json

python tests/mock_server.py &

agentperf-replay \
    --mode closed_loop \
    --concurrency 2 \
    --trace smoke_trace.json \
    --base-url http://localhost:8765 \
    --framework vllm \
    --model mock-model \
    --precision bf16
```

---

## Structure

```
agent-perf/
├── agentperf/
│   ├── clients.py          # HTTP client layer; streams tokens from OpenAI-compatible servers
│   ├── collectors.py       # GPU telemetry and server-metrics polling
│   ├── launch.py           # Framework process lifecycle (start, health-check, stop)
│   ├── manifest.py         # Run manifest creation and serialization
│   ├── metrics.py          # Stateless metric functions: TTFT, ITL, E2E, goodput, cache hit rate
│   ├── models.py           # Pydantic data models shared across the whole package
│   ├── replay.py           # Trace replay engine (closed-loop and open-loop)
│   └── report.py           # Report generation from Parquet result files
├── chapters/
│   ├── ch1_frameworks/     # Framework comparison: vLLM vs SGLang vs TRT-LLM
│   ├── ch2_quantization/   # Precision ladder: BF16 -> FP8 -> NVFP4
│   ├── ch3_scheduler_knobs/# vLLM one-knob sweeps (prefix cache, chunked prefill, max-seqs, GPU mem)
│   └── ch4_tensor_parallel/# TP=2 BF16 vs FP8 single-GPU on PCIe Gen5 without NVLink (70B)
├── quality/
│   ├── scorer.py           # Task quality scorer (exact match, function call, code exec)
│   └── task_pack.json      # Evaluation task suite for pass-rate measurement
├── quantize/
│   ├── fp8_recipe.py       # FP8 quantization recipe
│   └── nvfp4_recipe.py     # NVFP4 quantization recipe (sm_120 FP4 tensor cores)
├── tests/
│   └── mock_server.py      # FastAPI mock server for smoke tests without GPU
├── traces/
│   ├── baseline_chat/      # Baseline chat trace builder
│   ├── derived/            # Derived trace utilities
│   ├── synthetic/
│   │   └── generator.py    # Synthetic trace generator with named presets
│   └── schema.json         # Trace JSON schema (schema_version 1)
├── writeup/                # Analysis notebooks and draft writeup
├── pyproject.toml
└── METHODOLOGY.md          # Eight rules followed for every reported result
```

---

## The Four Chapters

### Chapter 1 — Frameworks under agent traffic

Compares vLLM, SGLang, and TensorRT-LLM on Llama-3.1-8B-Instruct (BF16, single GPU) across
four trace types that span the agent traffic spectrum: `baseline_chat`, `agent_shallow`,
`agent_deep`, and `agent_swarm`. Sweeps concurrency from 1 to 128. The central questions are
which framework extracts the most goodput from prefix KV reuse on deep-context agent traces,
and whether SGLang's radix-cache architecture yields a measurable TTFT advantage on high
shared-prefix workloads.

### Chapter 2 — The precision ladder on agent workloads

Runs both Llama-3.1-8B-Instruct and Llama-3.3-70B-Instruct through the BF16 → FP8 → NVFP4
precision ladder, measuring throughput, latency, and task pass rate at each step. Each
non-baseline config changes exactly one variable (precision) relative to its size-class
baseline. Quality is scored with `agentperf-score` over `quality/task_pack.json` after every
run, enabling a throughput-vs-quality trade-off analysis specific to multi-turn agent sessions.

### Chapter 3 — Scheduler and memory knobs

Holds model, framework (vLLM), and GPU fixed, then varies one knob per run: prefix caching
on/off, chunked prefill on/off, `--max-num-seqs` (16 / 32 / 64 / 128 / 256), and
`--gpu-memory-utilization` (0.60 / 0.75 / 0.90). Tests on `agent_deep` (20 sessions, 32 KB
shared system prompt, 50% tool-call ratio) and `agent_swarm` (100 sessions, short bursts,
20% tool-call ratio) to surface interactions that first-order intuition misses.

### Chapter 4 — Tensor parallelism over PCIe Gen5 without NVLink

Answers whether FP8 quantization on a single GPU dominates TP=2 BF16 tensor parallelism for
70B-scale agent serving when NVLink is unavailable. The comparison holds the model
(Llama-3.3-70B-Instruct) and endpoint fixed; the only axes of variation are precision and
number of GPUs. Includes NCCL all-reduce bandwidth profiling and nsys CUDA traces to
characterize the fraction of TP=2 wall time attributed to PCIe communication versus compute.

---

## Key Metrics

| Metric | Definition | SLO |
|---|---|---|
| TTFT (ms) | Time from request send to first token received | < 1000 ms |
| ITL (ms/tok) | Mean inter-token latency: (last_token − first_token) / (output_tokens − 1) | < 50 ms |
| Goodput (tok/s) | Output tokens per second for requests meeting both TTFT and ITL SLOs | — |
| Prefix Cache Hit Rate | Fraction of prompt tokens served from KV cache | — |
| Task Success Rate | Pass rate on `quality/task_pack.json` (exact match, function call, code exec) | — |

All latency measurements use `time.monotonic_ns()` on the client. Values reported as p50 and
p99 across compliant (HTTP 200) requests. Error rate (HTTP non-200 fraction) is tracked
separately and excluded from latency distributions.

See `METHODOLOGY.md` for the full eight-rule protocol followed for every reported result.

---

## Reproducing Results

Each chapter provides a self-contained run script:

```
chapters/ch1_frameworks/run.sh
chapters/ch2_quantization/run.sh
chapters/ch3_scheduler_knobs/run.sh
chapters/ch4_tensor_parallel/run.sh
```

Results are written to `results/<config_name>/run{1,2,3}/` as Parquet files plus JSON manifests.
Use `agentperf-report` to generate summary tables and plots from any results directory.

A local GPU matching the hardware specification above is required for chapters 1–4. See
`ENVIRONMENTS.md` for driver, CUDA, and framework version setup.

---

## Methodology

See `METHODOLOGY.md`. Eight rules, followed for every reported result:

1. Clock frequency locked with `nvidia-smi -lgc` before server start; released on exit.
2. Sixty-second warmup on the target trace before each measured window.
3. Three measured runs per configuration to assess variance.
4. One variable changed at a time relative to a documented baseline.
5. Trace checksums recorded in the run manifest to guarantee reproducibility.
6. Quality scored from the same task pack for every precision configuration.
7. GPU telemetry (utilization, power, VRAM) sampled throughout each run.
8. All raw Parquet files and manifests retained alongside reported aggregates.
