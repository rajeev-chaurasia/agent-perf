# Measurement Methodology

## SLO Definitions

| Metric | SLO |
|--------|-----|
| TTFT | 1000 ms |
| ITL | 50 ms/token (mean inter-token, excluding first token) |
| Goodput | output tok/s for requests meeting BOTH SLOs simultaneously |

---

## Eight Rules

### 1. Locked Clocks

Run `sudo nvidia-smi -lgc <min>,<max>` before every measured window. Record the locked frequency in the run manifest. Restore with `sudo nvidia-smi -rgc` after the run. Any run without a confirmed lock entry in the manifest is invalid.

### 2. Warmup

Execute at least 60 seconds of traffic before the measured window begins, using the identical configuration (model, batch size, concurrency, mode). Warmup results are discarded entirely and never included in reported distributions. The warmup period allows GPU caches, KV-cache allocators, and scheduler queues to reach steady state.

### 3. Multiple Runs

Collect at least 3 independent runs per configuration. Report the median as the point estimate. Plot min/max as whiskers on all figures. A claimed effect is valid only if it exceeds the observed run-to-run spread; if the whiskers of two configurations overlap, the result is inconclusive.

### 4. Pinned Versions

Every manifest must record:
- Inference framework name, commit hash or release tag
- Model identifier and Hugging Face model hash (sha256 of config + weights)
- CUDA version (`nvcc --version`)
- GPU driver version (`nvidia-smi --query-gpu=driver_version`)

Manifests without all four fields are rejected. Version pinning is non-negotiable for reproducibility across machines and time.

### 5. One Variable Per Comparison

Exactly one parameter changes between any two configurations being compared. All other parameters — batch size, sequence length, concurrency, quantization, tensor-parallel degree, sampling settings — are held constant. Multi-variable diffs are not reportable; they require decomposition into separate single-variable comparisons.

### 6. Mode Labeling

Open-loop and closed-loop results are never mixed in the same plot or table. Every figure, table, and data file is stamped with its mode (`open-loop` or `closed-loop`). Aggregates that silently combine modes are invalid regardless of how the numbers look.

### 7. Failure Accounting

Every request dispatched during a measured window must be accounted for. Successful and failed requests are both written to parquet with HTTP/gRPC status codes, error messages, and timestamps. Error rate is reported alongside every latency and throughput metric. Dropping or silently discarding failed requests to improve reported numbers is prohibited.

### 8. Reproducibility

Each experiment chapter lives under `chapters/chN/`. The file `chapters/chN/run.sh` must be a self-contained script that regenerates all results for that chapter from scratch: environment setup, server launch, traffic generation, and artifact collection. No undocumented manual steps. A result that cannot be reproduced by running `run.sh` on a machine meeting the hardware spec is not a result.

---

## Replay Modes

**Closed-loop:** M concurrent sessions run sequentially. Turn N+1 is sent only after turn N completes plus an optional think-time delay. Concurrency is bounded and controlled. Use closed-loop to characterize latency distributions at a fixed concurrency level.

**Open-loop:** Sessions arrive at a fixed inter-arrival rate independent of whether prior sessions have completed. The server accumulates backlog if it cannot keep pace. Use open-loop to find saturation throughput and observe queue buildup behavior.

Never report a single throughput figure without specifying which mode produced it.

---

## Client Overhead

The benchmarking client runs on the same machine as the inference server. CPU utilization of the client process is sampled throughout the measured window and recorded in the manifest. Any run where the client process sustains CPU% > 50% must be flagged as potentially client-bottlenecked; conclusions drawn from such runs require independent verification with a remote client before they are reportable.

---

## Validity Criterion

A finding is reportable if and only if all three conditions hold:

1. The result is consistent in direction and magnitude across all >= 3 runs.
2. The effect size exceeds the run-to-run variance (min/max whiskers of compared configurations do not overlap).
3. All 8 rules above were followed for every run contributing to the finding.

Findings that meet conditions 1 and 2 but violated any rule are noted as preliminary and require a clean re-run before publication.
