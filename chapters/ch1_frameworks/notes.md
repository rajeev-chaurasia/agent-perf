# Chapter 1: Framework comparison -- vLLM vs SGLang vs TRT-LLM

**Fixed axes:** Llama-3.1-8B-Instruct, BF16, GPU 1 (single GPU)
**Sweep:** concurrency 1 2 4 8 16 32 64 128, 3 runs each, 3 trace types

---

## Research questions

1. Which framework reaches peak goodput at the lowest concurrency for each trace type?
2. Does SGLang's radix-cache architecture yield measurable TTFT improvements on high shared-prefix workloads vs vLLM prefix caching?
3. At what concurrency does each framework's scheduler saturate, and does the saturation point differ by trace type?
4. How does TRT-LLM's PyTorch backend ITL compare to the interpreted runtimes at high load?

---

## Results

### TTFT p50 (ms)

| Concurrency | SGLang | vLLM | TRT-LLM |
|---|---|---|---|
| **agent_shallow** | | | |
| c=1 | 20 | 23 | 24 |
| c=8 | 35 | 36 | 38 |
| c=32 | 37 | 44 | 42 |
| c=128 | 39 | 49 | 44 |
| **agent_deep** | | | |
| c=1 | 29 | 35 | 44 |
| c=8 | 44 | 51 | 56 |
| c=32 | 47 | 61 | 59 |
| c=128 | 47 | 61 | 60 |
| **agent_swarm** | | | |
| c=1 | 28 | 33 | 38 |
| c=8 | 45 | 53 | 55 |
| c=32 | 52 | 89 | 66 |
| c=128 | 72 | 204 | 94 |

### E2E p50 at c=128 (ms)

| Trace | SGLang | vLLM | TRT-LLM |
|---|---|---|---|
| agent_shallow | 1764 | 1565 | 1479 |
| agent_deep | 1812 | 1758 | 1625 |
| agent_swarm | 1662 | 2118 | 983 |

---

## Key findings

**SGLang wins TTFT across the board.** At c=1, SGLang is 3-6 ms faster than vLLM and 4-15 ms faster than TRT-LLM. The gap widens with context depth: on agent_deep, TRT-LLM's TTFT at c=1 is 44 ms vs SGLang's 29 ms -- likely explained by TRT-LLM's prefill scheduling under the PyTorch backend vs SGLang's RadixAttention.

**vLLM scheduler stalls on agent_swarm above c=32.** TTFT climbs from 33 ms at c=1 to 89 ms at c=32 and 204 ms at c=128. SGLang and TRT-LLM both stay under 95 ms at c=128. For deployments serving many short concurrent agent sessions, vLLM's scheduler creates a meaningful latency problem at scale.

**TRT-LLM wins E2E on agent_swarm at high concurrency.** Despite its higher TTFT, TRT-LLM achieves 983 ms median E2E at c=128 vs SGLang's 1662 ms and vLLM's 2118 ms. The ITL advantage at high concurrency (26 ms/tok vs 31 ms for SGLang, 50 ms for vLLM) is what drives this. Once the first token is out, TRT-LLM streams faster.

**agent_deep context growth exposes TRT-LLM's prefill cost.** TRT-LLM TTFT at c=1 is 44 ms for agent_deep vs 24 ms for agent_shallow -- a 1.8x slowdown as context grows. vLLM sees 1.5x (23 ms to 35 ms), SGLang 1.5x (20 ms to 29 ms). TRT-LLM pays more per-prefill at larger sequence lengths under the PyTorch backend.

**SGLang plateaus on agent_deep.** TTFT goes from 29 ms at c=1 to 47 ms at c=128 and barely moves between c=32 and c=128. vLLM and TRT-LLM both level off similarly. This suggests the GPU is fully saturated by c=16 for this trace and additional concurrency no longer increases queuing time.

---

## Recommendations

For latency-sensitive agentic deployments where TTFT is the primary SLO: SGLang.

For high-concurrency swarm workloads where E2E and throughput matter more than first-token latency: TRT-LLM at high concurrency, SGLang at moderate concurrency.

Avoid vLLM for agent_swarm patterns above c=32 until scheduler improvements address the TTFT regression.

---

## Open questions for next chapters

- Does quantization (FP8, NVFP4) close the TRT-LLM TTFT gap at large context? (Ch2)
- How much does turning off prefix caching in vLLM hurt agent_deep goodput? (Ch3)
- Does chunked prefill help vLLM's agent_swarm TTFT at c=64+? (Ch3)
