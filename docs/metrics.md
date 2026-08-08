# Metrics

All latency measurements are taken on the client side using `time.monotonic_ns()`. The monotonic clock is used rather than wall time to avoid sensitivity to NTP adjustments during a run. Timestamps are recorded in nanoseconds and converted to milliseconds for reporting. No server-side timing is used; the numbers reflect what a real client would observe.

Each measurement corresponds to a single request (one turn in one session). The Parquet output contains one row per turn, with the computed latency columns alongside raw token counts and HTTP status. Aggregated statistics (p50, p99 across runs) are computed in `agentperf/metrics.py` using the `percentile()` function, which filters out the -1.0 sentinel value used for failed or incomplete requests before computing numpy percentiles.

## TTFT

Time-to-first-token is `(first_token_ns - request_sent_ns) * 1e-6` ms, where `first_token_ns` is the timestamp recorded when the first non-empty content chunk arrives in the SSE stream. It covers the full round trip: network transit to the server, time spent waiting in the request queue, prefill compute, and network transit back to the first byte. At concurrency 1, TTFT is dominated by prefill time. At higher concurrency, queuing delay becomes the dominant term.

Requests where `http_status != 200` or where no content chunk was ever received return a sentinel value of -1.0 for TTFT. These are excluded from all percentile computations.

## ITL

Inter-token latency is `(last_token_ns - first_token_ns) / (output_tokens - 1)` ms. It measures the mean time between successive tokens during the decode phase of a single request. The denominator is `output_tokens - 1` rather than `output_tokens` because the gap count between N tokens is N-1. A request that produces fewer than 2 tokens cannot have an inter-token gap and returns -1.0. Requests where the stream did not complete also return -1.0.

ITL is sensitive to decode batching efficiency. When many requests are in the decode phase simultaneously, the GPU batches their KV lookups together; ITL reflects how well that batching amortizes per-token overhead. A rising ITL at high concurrency typically indicates that the batch size is large enough to create memory bandwidth pressure rather than compute pressure.

## E2E Latency

End-to-end latency is `(last_token_ns - request_sent_ns) * 1e-6` ms. It covers the entire request lifetime from the client's perspective: queuing, prefill, decode, and any streaming overhead. E2E latency is the most direct measure of user-perceived response time for a complete reply. It returns -1.0 if the stream did not complete successfully.

## Goodput

Goodput counts only the output tokens from requests that satisfy both the TTFT SLO (default: TTFT < 1000 ms) and the ITL SLO (default: ITL < 50 ms), then divides by the wall-clock window of the run in seconds. Requests that produce fewer than 2 tokens pass the ITL check automatically (ITL returns -1.0 for short responses, which is treated as compliant). This gives a throughput figure that reflects what the system can actually deliver under quality-of-service constraints, rather than raw token throughput that ignores whether responses arrived in time.

Goodput = (sum of output_tokens for SLO-compliant requests) / (wall_time_s)

The wall-clock window is derived from the maximum `e2e_ms` value in the run, converted to seconds. This is conservative; it includes any tail latency from slow requests, which tends to understate goodput slightly at high concurrency.

## Error Rate

Error rate is the fraction of turns where `http_status != 200`. These requests are excluded from all latency statistics. An error rate above a few percent during a measurement run indicates the server was overwhelmed or misconfigured, and the latency numbers for that run should be treated with caution.

## Reporting Convention

All results in this repository are reported as p50 and p99 across the three independent runs for each (framework, trace, concurrency) combination. The three runs are independent replays against a freshly started server, with a 60-second warmup period before each measurement. Reporting p50 captures typical behavior; p99 captures tail latency, which is what matters most for interactive agent workloads where a single slow turn stalls the entire session.
