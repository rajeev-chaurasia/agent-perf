"""Metric collectors: GPU sampler and Prometheus scraper.

Two independent components with no shared state.
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import httpx

from agentperf.models import GpuSample, ServerMetrics

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# GPU Sampler
# ---------------------------------------------------------------------------

_NVIDIA_SMI_CMD: list[str] = [
    "nvidia-smi",
    "--query-gpu=index,utilization.gpu,power.draw,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]

_MIB_TO_GB: float = 1.0 / 1024.0


class GpuSampler:
    """Background thread that polls nvidia-smi at a fixed interval.

    Usage::

        with GpuSampler(gpu_ids=[0, 1]) as sampler:
            # ... run workload ...
            readings = sampler.samples
    """

    def __init__(self, gpu_ids: list[int], interval_ms: int = 100) -> None:
        self._gpu_ids: frozenset[int] = frozenset(gpu_ids)
        self._interval_s: float = interval_ms / 1000.0
        self._deque: collections.deque[GpuSample] = collections.deque(maxlen=10_000)
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._run,
            name="GpuSampler",
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GpuSampler":
        self._stop_event.clear()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop_event.set()
        self._thread.join()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def samples(self) -> list[GpuSample]:
        """Return a snapshot of all collected samples."""
        return list(self._deque)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Thread target: poll nvidia-smi until stop event is set."""
        while not self._stop_event.is_set():
            ts_ms = time.time() * 1000.0
            self._collect(ts_ms)
            self._stop_event.wait(timeout=self._interval_s)

    def _collect(self, ts_ms: float) -> None:
        """Run nvidia-smi once and append parsed samples to the deque."""
        try:
            result = subprocess.run(
                _NVIDIA_SMI_CMD,
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                sample = self._parse_line(line.strip(), ts_ms)
                if sample is not None:
                    self._deque.append(sample)
        except Exception as exc:  # noqa: BLE001
            print(f"[GpuSampler] nvidia-smi error: {exc}", file=sys.stderr)

    def _parse_line(self, line: str, ts_ms: float) -> GpuSample | None:
        """Parse one CSV row from nvidia-smi output.

        Expected columns (nounits): index, util%, power_w, mem_used_mib, mem_total_mib
        """
        if not line:
            return None
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None
        try:
            gpu_id = int(parts[0])
            if gpu_id not in self._gpu_ids:
                return None
            util_pct = float(parts[1])
            power_w = float(parts[2])
            vram_used_mib = float(parts[3])
            return GpuSample(
                ts_ms=ts_ms,
                gpu_id=gpu_id,
                util_pct=util_pct,
                power_w=power_w,
                vram_used_gb=vram_used_mib * _MIB_TO_GB,
            )
        except (ValueError, IndexError) as exc:
            print(f"[GpuSampler] parse error on line {line!r}: {exc}", file=sys.stderr)
            return None


# ---------------------------------------------------------------------------
# Prometheus metrics scraper
# ---------------------------------------------------------------------------

# Matches: metric_name{optional_labels} numeric_value
_PROM_LINE_RE: re.Pattern[str] = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)

# Ordered preference: first match wins for each ServerMetrics field.
_CACHE_HIT_RATE_KEYS: tuple[str, ...] = (
    "vllm:cache_hit_rate",
    "vllm:prefix_cache_hit_rate",
)
_GPU_CACHE_USAGE_KEYS: tuple[str, ...] = (
    "vllm:gpu_cache_usage_perc",
    "sglang:token_usage",
)
_NUM_RUNNING_KEYS: tuple[str, ...] = (
    "vllm:num_requests_running",
    "sglang:num_running_reqs",
)

_ALL_INTERESTING: frozenset[str] = frozenset(
    _CACHE_HIT_RATE_KEYS + _GPU_CACHE_USAGE_KEYS + _NUM_RUNNING_KEYS
)


class MetricsScraper:
    """Async scraper for a Prometheus /metrics endpoint.

    Usage::

        async with httpx.AsyncClient() as client:
            scraper = MetricsScraper("http://localhost:8000/metrics")
            metrics = await scraper.scrape(client)
    """

    def __init__(self, metrics_url: str) -> None:
        self._metrics_url: str = metrics_url

    async def scrape(self, client: httpx.AsyncClient) -> ServerMetrics:
        """Fetch and parse the Prometheus metrics endpoint.

        Returns ``ServerMetrics()`` (all-None) on any error; never raises.
        """
        try:
            response = await client.get(self._metrics_url, timeout=5.0)
            response.raise_for_status()
            raw = self._parse_prometheus(response.text)
            return self._build_server_metrics(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"[MetricsScraper] scrape error ({self._metrics_url}): {exc}", file=sys.stderr)
            return ServerMetrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_prometheus(text: str) -> dict[str, float]:
        """Extract interesting metric values from Prometheus text format."""
        extracted: dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _PROM_LINE_RE.match(line)
            if match is None:
                continue
            name, value_str = match.group(1), match.group(2)
            if name in _ALL_INTERESTING:
                try:
                    extracted[name] = float(value_str)
                except ValueError:
                    pass
        return extracted

    @staticmethod
    def _first(raw: dict[str, float], keys: tuple[str, ...]) -> float | None:
        """Return the value of the first matching key, or None."""
        for key in keys:
            if key in raw:
                return raw[key]
        return None

    def _build_server_metrics(self, raw: dict[str, float]) -> ServerMetrics:
        cache_hit_rate = self._first(raw, _CACHE_HIT_RATE_KEYS)
        gpu_cache_usage_perc = self._first(raw, _GPU_CACHE_USAGE_KEYS)
        num_running_float = self._first(raw, _NUM_RUNNING_KEYS)
        num_running: int | None = (
            int(num_running_float) if num_running_float is not None else None
        )
        return ServerMetrics(
            cache_hit_rate=cache_hit_rate,
            gpu_cache_usage_perc=gpu_cache_usage_perc,
            num_running_requests=num_running,
            raw=dict(raw),
        )
