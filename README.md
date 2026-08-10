# agent-perf

Reproducible benchmarks for LLM inference serving under agentic workloads. Measures TTFT, ITL, E2E latency, and goodput across vLLM, SGLang, and TensorRT-LLM on a 96 GB workstation GPU (sm_120), focused on multi-turn agent traffic patterns.

Standard chat benchmarks (ShareGPT throughput, single-turn TTFT) miss three things that matter for agents: deep multi-turn sessions that grow KV cache pressure over time, bursty tool-call turns with short outputs, and many concurrent sessions racing for cache capacity. This suite measures all three.

---

## Hardware

| | |
|---|---|
| GPUs | 2x 96 GB workstation GPUs (sm_120, 188 SMs) |
| Interconnect | PCIe Gen5, no NVLink |
| CUDA | 13.2, driver 595.84 |
| GPU assignment | GPU 1: benchmark (all chapters); GPU 0: display, also used for TP=2 in Ch4 |

---

## Quick start

No GPU needed for the smoke test:

```bash
pip install -e ".[dev]"

# Generate a trace
agentperf-generate --preset agent_shallow --output smoke.json

# Start mock server
python tests/mock_server.py &

# Run replay
agentperf-replay \
    --mode closed_loop \
    --concurrency 4 \
    --trace smoke.json \
    --base-url http://localhost:8765 \
    --framework vllm \
    --model mock-model \
    --precision bf16
```

To run a full chapter benchmark (GPU required):

```bash
bash chapters/ch1_frameworks/run.sh
```

See [docs/setup.md](docs/setup.md) for framework venv setup and model download.

---

## Chapter 1 results -- Framework comparison

**Config:** Llama-3.1-8B-Instruct, BF16, single GPU, concurrency 1-128, 3 runs per point.

### TTFT p50 at c=1 (single session baseline)

| Framework | agent_shallow | agent_deep | agent_swarm |
|---|---|---|---|
| SGLang | **20 ms** | **29 ms** | **28 ms** |
| vLLM | 23 ms | 35 ms | 33 ms |
| TRT-LLM | 24 ms | 44 ms | 38 ms |

SGLang is 10-15 ms faster to first token across every workload type at baseline concurrency.

### TTFT p50 at c=128 (high concurrency)

| Framework | agent_shallow | agent_deep | agent_swarm |
|---|---|---|---|
| SGLang | 39 ms | 47 ms | 72 ms |
| vLLM | 49 ms | 61 ms | **204 ms** |
| TRT-LLM | 44 ms | 60 ms | 94 ms |

vLLM's scheduler stalls above c=32 on agent_swarm: TTFT grows from 33 ms to 204 ms (6x). SGLang holds at 72 ms (2.5x). For a swarm of 64+ parallel agents, vLLM produces noticeably worse first-response latency.

### E2E latency at c=128, agent_swarm

| Framework | E2E p50 |
|---|---|
| **TRT-LLM** | **983 ms** |
| SGLang | 1662 ms |
| vLLM | 2118 ms |

TRT-LLM streams tokens faster once past the first token -- lower ITL at high load. For long streamed responses at high concurrency, TRT-LLM has the edge.

Full plots and per-metric breakdowns: [docs/framework-comparison.md](docs/framework-comparison.md)

---

## Structure

```
agent-perf/
├── agentperf/
│   ├── clients.py        httpx streaming client, timestamp capture
│   ├── manifest.py       run manifest with trace checksum and env snapshot
│   ├── metrics.py        stateless metric functions (TTFT, ITL, E2E, goodput)
│   ├── models.py         Pydantic data models shared across the package
│   ├── replay.py         trace replay engine (closed-loop and open-loop)
│   └── report.py         plot and markdown table generation
├── chapters/
│   ├── ch1_frameworks/   vLLM vs SGLang vs TRT-LLM
│   ├── ch2_quantization/ BF16 -> FP8 -> NVFP4 precision ladder
│   ├── ch3_scheduler_knobs/  vLLM knob sweeps (prefix cache, chunked prefill, max-seqs)
│   └── ch4_tensor_parallel/  TP=2 BF16 vs FP8 single-GPU on PCIe Gen5 (70B)
├── docs/                 architecture, setup, metrics reference, results
├── quality/              task quality scorer and evaluation pack
├── quantize/             FP8 and NVFP4 quantization recipes
├── tests/                mock server for offline smoke tests
├── traces/               trace schema and synthetic generator
├── scripts/              analysis and plot generation scripts
├── ENVIRONMENTS.md       hardware specs, framework versions, model checksums
└── METHODOLOGY.md        eight measurement rules followed for every result
```

---

## Trace types

Three workload shapes cover the agent traffic spectrum:

**agent_shallow** -- 50 sessions, ~7 turns each, short back-and-forth exchanges. Closest to a standard chatbot but with multi-turn context.

**agent_deep** -- 20 sessions, ~23 turns each, with context that grows as tool results accumulate. By turn 15, prompts reach 8-30k tokens. Designed to stress KV cache reuse under sustained context growth.

**agent_swarm** -- 100 sessions, 3 turns each, all short. Models a fleet of agents running independent tasks in parallel. Tests scheduler fan-out and TTFT stability under high session count.

---

## Metrics

| Metric | Definition |
|---|---|
| TTFT | Time from request send to first token received (ms) |
| ITL | Mean inter-token latency: (last - first token) / (output tokens - 1) (ms) |
| E2E | Total wall time from request send to last token (ms) |
| Goodput | Output tokens/s for requests meeting TTFT < 1000 ms and ITL < 50 ms |

All timestamps use `time.monotonic_ns()` on the client. Stats reported as p50 and p99 across 3 measured runs after a 60-second warmup.

---

## Reproducing results

Each chapter has a self-contained run script:

```bash
chapters/ch1_frameworks/run.sh
chapters/ch2_quantization/run.sh
chapters/ch3_scheduler_knobs/run.sh
chapters/ch4_tensor_parallel/run.sh
```

Results write to `chapters/<chapter>/results/` as Parquet files plus JSON run manifests. Generate a report from any results directory:

```bash
agentperf-report \
    --parquet chapters/ch1_frameworks/results/vllm/agent_deep/**/*.parquet \
    --labels ... \
    --output-dir chapters/ch1_frameworks/results/report \
    --plot all
```

See [METHODOLOGY.md](METHODOLOGY.md) for the eight measurement rules followed for every reported result.

---

## Coming chapters

| Chapter | Topic | Status |
|---|---|---|
| Ch2 | Precision ladder: BF16, FP8, NVFP4 on 8B and 70B | Planned |
| Ch3 | vLLM scheduler knobs: prefix cache, chunked prefill, max-seqs, GPU mem | Planned |
| Ch4 | TP=2 BF16 vs single-GPU FP8 on PCIe Gen5 without NVLink (70B) | Planned |
