# Chapter 4: Tensor Parallelism — 70B BF16 TP=2 vs FP8 Single GPU over PCIe Gen5

## Research Question

Does FP8 quantization on a single GPU dominate TP=2 BF16 tensor parallelism for 70B-scale
agent serving over PCIe Gen5 when NVLink is unavailable?

Sub-questions:

- At what concurrency level (if any) does PCIe Gen5 all-reduce latency become the throughput
  bottleneck for TP=2?
- Does the FP8 memory bandwidth advantage on a single GPU translate to lower TTFT and ITL
  across all concurrency tiers, or only at low concurrency?
- How does KV-cache capacity differ between the two configs (one GPU @ FP8 vs two GPUs @
  BF16), and at what context length does the capacity gap materialize as a quality cliff?
- What does the nsys CUDA trace reveal about the fraction of TP=2 wall time spent in NCCL
  all-reduce vs attention kernels?

## Interconnect Baseline

<!-- OWNER WRITES: paste nccl_bandwidth.txt summary here (bus bandwidth at 1G message size,
     and the bandwidth at the inflection point where latency starts to dominate) -->

## Config Comparison

| Field           | fp8_single (base)             | bf16_tp2 (comparison)         |
|-----------------|-------------------------------|-------------------------------|
| precision       | fp8                           | bf16                          |
| gpu_ids         | [1]                           | [0, 1]                        |
| extra_args      | {}                            | {"--tensor-parallel-size":"2"}|
| model           | Llama-3.3-70B-Instruct        | Llama-3.3-70B-Instruct        |
| base_url        | http://localhost:8000         | http://localhost:8000         |

The two configs share the same model and endpoint; the only axes of variation are the
precision/quantization strategy and the number of GPUs (which are coupled for this comparison).

## Results Skeleton

### NCCL All-Reduce Bandwidth (PCIe Gen5, no NVLink)

<!-- OWNER WRITES: bus bandwidth numbers from results/nccl_bandwidth.txt, e.g.
     - 1 KB message:  X GB/s
     - 1 GB message:  X GB/s
     Key finding: [bottleneck region and effective bandwidth ceiling] -->

### TTFT (ms) — agent_deep trace, closed_loop

| Concurrency | fp8_single p50 | fp8_single p99 | bf16_tp2 p50 | bf16_tp2 p99 |
|-------------|----------------|----------------|--------------|--------------|
| 1           | <!-- -->        | <!-- -->        | <!-- -->      | <!-- -->      |
| 4           | <!-- -->        | <!-- -->        | <!-- -->      | <!-- -->      |
| 16          | <!-- -->        | <!-- -->        | <!-- -->      | <!-- -->      |
| 64          | <!-- -->        | <!-- -->        | <!-- -->      | <!-- -->      |

<!-- OWNER WRITES: fill from results/fp8_single/run*/  and results/bf16_tp2/run*/ parquet files -->

### ITL (ms) — inter-token latency

<!-- OWNER WRITES: same table structure as TTFT above -->

### Throughput (tok/s)

<!-- OWNER WRITES: total output tokens / wall-clock seconds for each config x concurrency cell -->

### GPU Utilization & Power

<!-- OWNER WRITES: mean GPU util % and power draw (W) for each config during the measured window -->

## Observations / Hypotheses

<!-- OWNER WRITES: qualitative observations after reviewing the data, e.g.
     - Which config wins at low concurrency vs high concurrency?
     - Does the PCIe Gen5 bandwidth appear in the nsys trace as idle time between all-reduce
       calls and the next compute kernel?
     - Any unexpected behavior (e.g. fp8 model load time, TP=2 warmup stall)? -->

## Follow-up Experiments

<!-- OWNER WRITES: ideas for next-iteration experiments based on results, e.g.
     - Test TP=2 BF16 with NVLink to isolate the PCIe tax
     - Sweep prefix_kb in the trace generator to stress KV-cache capacity per config
     - Profile FP8 dequantization overhead with nsys at concurrency=64 -->
