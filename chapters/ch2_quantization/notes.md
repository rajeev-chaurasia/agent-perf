# Chapter 2: Quantization — BF16 vs FP8 vs NVFP4

## Research Question

What does NVFP4 cost in task quality and gain in throughput on sm_120?
At what model size does the throughput gain outweigh the quality regression,
and does FP8 offer a better quality/throughput trade-off than NVFP4 for
agent workloads with long multi-turn context?

## Configs Covered

| Config file         | Model                              | Precision | GPUs    | Notes           |
|---------------------|------------------------------------|-----------|---------|-----------------|
| 8b_bf16.json        | Llama-3.1-8B-Instruct              | BF16      | [1]     | 8B baseline     |
| 8b_fp8.json         | Llama-3.1-8B-Instruct              | FP8       | [1]     | 1 var vs base   |
| 8b_nvfp4.json       | Llama-3.1-8B-Instruct              | NVFP4     | [1]     | 1 var vs base   |
| 70b_bf16_tp2.json   | Llama-3.3-70B-Instruct             | BF16      | [0,1]   | 70B baseline TP=2 |
| 70b_fp8.json        | Llama-3.3-70B-Instruct             | FP8       | [1]     | 1 var vs base   |
| 70b_nvfp4.json      | Llama-3.3-70B-Instruct             | NVFP4     | [1]     | 1 var vs base   |

Each non-baseline config changes exactly one variable relative to its size-class
baseline (precision only); all other fields are held constant.

## Experimental Protocol

- Clock lock (`nvidia-smi -lgc`) applied before server start; released in `trap EXIT ERR`.
- 60-second warmup on `agent_deep` trace before each measured window.
- Three measured runs per config (`for run in 1 2 3`) to assess variance.
- Quality scored with `agentperf-score` over `quality/evaluation-tasks.json` after all runs.
- Results land in `results/<config_name>/run{1,2,3}/`.

## Open Questions

1. Does NVFP4 KV-cache reduction translate to proportionally higher concurrency
   capacity, or does compute become the bottleneck first?
2. How does FP8 quality compare to BF16 on multi-hop tool-call sequences in
   `agent_deep` — do reasoning errors compound across turns?
3. Is the NVFP4 quality gap larger for 70B than 8B (smaller models may have
   less redundancy to absorb quantization noise)?
4. What is the effective tokens-per-second-per-GPU gain of NVFP4 vs FP8 on
   H100 SXM vs B200 — does the sm_120 FP4 tensor core advantage hold at
   real agent batch sizes?

## Findings

<!-- OWNER WRITES: precision ladder table after running -->

<!-- OWNER WRITES: throughput (tok/s) vs quality (pass rate) scatter plot -->

<!-- OWNER WRITES: p50/p95 TTFT and ITL breakdown per precision -->

<!-- OWNER WRITES: notes on calibration dataset used for FP8/NVFP4 quantization -->

<!-- OWNER WRITES: GPU VRAM headroom numbers per config (enables batch-size tuning) -->
