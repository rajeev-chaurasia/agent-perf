# Chapter 3 — vLLM Scheduler Knob Sweeps

**Fixed**: vLLM, Llama-3.1-8B-Instruct FP8, GPU 1  
**Traces**: `agent_deep` (20 sessions, deep multi-turn, 50% tool-call), `agent_swarm` (100 sessions, short bursts, 20% tool-call)

## Research Questions

1. **Prefix caching vs. agent traffic** — agent sessions share a large system prompt (32 KB).
   Does prefix caching visibly reduce TTFT on repeated turns? Does it matter more on `agent_deep`
   (long sessions with prefix reuse) or `agent_swarm` (many short sessions, less reuse)?

2. **Chunked prefill interaction** — chunked prefill splits long prefills across multiple
   scheduler steps. How does enabling it interact with prefix caching? Does it hurt or help
   TTFT on the 32 KB shared-prefix traces?

3. **Max-seqs saturation point** — at what `--max-num-seqs` value does throughput plateau
   for agent traffic? Does the saturation point differ between `agent_deep` (long, deep queues)
   and `agent_swarm` (many short arrivals)?

4. **GPU memory vs. KV-cache capacity** — lower `--gpu-memory-utilization` shrinks the KV
   cache. For the 32 KB shared prefix, does cache eviction become a bottleneck at 0.60?
   Does 0.90 unlock meaningful extra throughput at the cost of OOM risk?

5. **Surprising interactions** — are there combinations of knobs where the first-order
   intuition breaks down (e.g., more seqs hurts throughput due to scheduler overhead,
   or prefix caching slows things down due to hash overhead at low concurrency)?

## Configs Tested

| Config file            | Changed knob              | Value (vs base=64 seqs, 0.75 mem) |
|------------------------|---------------------------|-----------------------------------|
| `base_8b_fp8.json`     | —                         | prefix_cache=on, chunked=off, seqs=64, mem=0.75 |
| `prefix_cache_off.json`| `--enable-prefix-caching` | false                             |
| `chunked_prefill_on.json` | `--enable-chunked-prefill` | true                           |
| `max_seqs_016.json`    | `--max-num-seqs`          | 16                                |
| `max_seqs_032.json`    | `--max-num-seqs`          | 32                                |
| `max_seqs_128.json`    | `--max-num-seqs`          | 128                               |
| `max_seqs_256.json`    | `--max-num-seqs`          | 256                               |
| `gpu_mem_060.json`     | `--gpu-memory-utilization`| 0.60                              |
| `gpu_mem_090.json`     | `--gpu-memory-utilization`| 0.90                              |

## Key Metrics to Compare

- **TTFT p50 / p95** — first-token latency; most sensitive to prefix-cache hits and chunked prefill
- **ITL p50 / p95** — inter-token latency; sensitive to scheduler batch size and concurrency
- **E2E p50 / p95** — total turn latency; summary metric across knobs
- **Cache hit rate** — from vLLM `/metrics` endpoint; validates prefix-caching effect
- **Requests/s** — overall throughput; primary signal for max-seqs and gpu-mem sweeps

## Results

<!-- OWNER WRITES: paste agentperf-report output or summary table here after running -->

### Prefix caching (base vs prefix_cache_off)

<!-- OWNER WRITES: TTFT delta, cache_hit_rate on agent_deep vs agent_swarm -->

### Chunked prefill (base vs chunked_prefill_on)

<!-- OWNER WRITES: TTFT and ITL deltas; note if interaction with prefix caching is unexpected -->

### max-num-seqs sweep (16 / 32 / 64 / 128 / 256)

<!-- OWNER WRITES: throughput and p95 latency table; note saturation point -->

### GPU memory utilization sweep (0.60 / 0.75 / 0.90)

<!-- OWNER WRITES: KV-cache eviction events (if any), throughput and latency deltas -->

## Takeaways

<!-- OWNER WRITES: 2-3 sentence summary of the most actionable findings for production deployments -->
