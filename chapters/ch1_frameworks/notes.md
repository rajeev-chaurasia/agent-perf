# Chapter 1: Framework Comparison — vLLM vs SGLang vs TRT-LLM

**Fixed axes:** Llama-3.1-8B-Instruct · BF16 · GPU 1 (single-GPU)
**Sweep:** concurrency 1 2 4 8 16 32 64 128 · 3 runs each · 4 trace types

---

## Research Questions

1. How much does prefix reuse (KV cache hit rate) improve goodput on deep-context agent traces (`agent_deep`, `agent_swarm`) versus short-context chat (`baseline_chat`)?
2. Which framework reaches peak goodput at the lowest concurrency level for each trace type?
3. Does SGLang's radix-cache architecture yield a measurable TTFT advantage on `agent_swarm` (high shared-prefix overlap) compared to vLLM prefix caching?
4. How does TRT-LLM engine compilation overhead and kernel fusion affect ITL distribution relative to interpreted runtimes?
5. At what concurrency does each framework saturate GPU utilization, and does the saturation point differ across trace types?

---

## Expected Plots

- `goodput_vs_concurrency.png` — 3 frameworks × 4 trace types; SLO: TTFT ≤ 1000 ms, ITL ≤ 50 ms
- `cache_hit_vs_depth.png` — KV cache hit rate by trace type per framework
- `ttft_vs_concurrency.png` — median and p99 TTFT curves

---

## Findings

<!-- OWNER WRITES: headline numbers after running run.sh -->

### vLLM

<!-- OWNER WRITES: peak goodput (tokens/s at SLO), cache hit rates per trace, saturation concurrency -->

### SGLang

<!-- OWNER WRITES: peak goodput, radix-cache hit rates per trace, saturation concurrency -->

### TRT-LLM

<!-- OWNER WRITES: peak goodput, KV cache hit rates, ITL distribution notes -->

---

## Observations

<!-- OWNER WRITES: cross-framework takeaways, anomalies, surprising results -->

---

## Next Steps / Open Questions

<!-- OWNER WRITES: hypotheses to test in Chapter 2 (quantization) or Chapter 3 (scheduler knobs) -->
