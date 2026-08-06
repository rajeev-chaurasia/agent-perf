"""Pure metric computation for agentperf.

Zero I/O, zero side effects. All functions are stateless transforms over
TurnResult lists and scalar inputs.
"""
from __future__ import annotations

import numpy as np

from agentperf.models import TurnResult

NS_TO_MS: float = 1e-6


def ttft_ms(results: list[TurnResult]) -> list[float]:
    """Time-to-first-token in milliseconds per request.

    Returns -1.0 for requests where http_status != 200 or first_token_ns == 0.
    """
    out: list[float] = []
    for r in results:
        if r.http_status != 200 or r.first_token_ns == 0:
            out.append(-1.0)
        else:
            out.append((r.first_token_ns - r.request_sent_ns) * NS_TO_MS)
    return out


def itl_ms(results: list[TurnResult]) -> list[float]:
    """Mean inter-token latency in milliseconds per request.

    Computed as (last_token_ns - first_token_ns) / (output_tokens - 1) in ms.
    Returns -1.0 if http_status != 200, first_token_ns == 0, last_token_ns == 0,
    or output_tokens < 2.

    Divides by (output_tokens - 1), NOT output_tokens.
    """
    out: list[float] = []
    for r in results:
        if (
            r.http_status != 200
            or r.first_token_ns == 0
            or r.last_token_ns == 0
            or r.output_tokens < 2
        ):
            out.append(-1.0)
        else:
            span_ms = (r.last_token_ns - r.first_token_ns) * NS_TO_MS
            out.append(span_ms / (r.output_tokens - 1))
    return out


def e2e_ms(results: list[TurnResult]) -> list[float]:
    """End-to-end latency in milliseconds per request.

    Measured from request_sent_ns to last_token_ns.
    Returns -1.0 if http_status != 200 or last_token_ns == 0.
    """
    out: list[float] = []
    for r in results:
        if r.http_status != 200 or r.last_token_ns == 0:
            out.append(-1.0)
        else:
            out.append((r.last_token_ns - r.request_sent_ns) * NS_TO_MS)
    return out


def output_throughput_tps(results: list[TurnResult], wall_time_s: float) -> float:
    """Total output throughput in tokens per second.

    Sums output_tokens across all successful (http_status == 200) results
    and divides by wall_time_s.
    Returns -1.0 if wall_time_s <= 0.
    """
    if wall_time_s <= 0:
        return -1.0
    total_tokens = sum(
        r.output_tokens for r in results if r.http_status == 200 and r.output_tokens >= 0
    )
    return total_tokens / wall_time_s


def goodput(
    results: list[TurnResult],
    wall_time_s: float,
    ttft_slo_ms: float = 1000.0,
    itl_slo_ms: float = 50.0,
) -> float:
    """Output tokens per second for requests meeting BOTH SLOs.

    A request meets the TTFT SLO when its ttft value is != -1.0 and <= ttft_slo_ms.
    A request meets the ITL SLO when its itl value is -1.0 (output_tokens < 2,
    i.e. short response) OR itl value is <= itl_slo_ms.

    Returns -1.0 if wall_time_s <= 0.
    """
    if wall_time_s <= 0:
        return -1.0

    ttfts = ttft_ms(results)
    itls = itl_ms(results)

    compliant_tokens = 0
    for r, t, i in zip(results, ttfts, itls):
        ttft_ok = t != -1.0 and t <= ttft_slo_ms
        itl_ok = i == -1.0 or i <= itl_slo_ms
        if ttft_ok and itl_ok and r.output_tokens >= 0:
            compliant_tokens += r.output_tokens

    return compliant_tokens / wall_time_s


def cache_hit_rate(server_metrics: dict) -> float:
    """Return cache_hit_rate from a server metrics dict.

    Returns -1.0 if the key is missing or its value is None.
    """
    value = server_metrics.get("cache_hit_rate")
    if value is None:
        return -1.0
    return float(value)


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile, ignoring -1.0 sentinel values.

    Returns -1.0 if no valid (non -1.0) values remain after filtering.
    """
    valid = [v for v in values if v != -1.0]
    if not valid:
        return -1.0
    return float(np.percentile(valid, p))


def error_rate(results: list[TurnResult]) -> float:
    """Fraction of results with http_status != 200.

    Returns 0.0 if results is empty.
    """
    if not results:
        return 0.0
    errors = sum(1 for r in results if r.http_status != 200)
    return errors / len(results)
