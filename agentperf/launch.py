"""
Framework process launchers for vLLM, SGLang, and TRT-LLM.

Each launcher encapsulates CLI command construction and server lifecycle
management for its respective framework. All servers are assumed to run
locally on the same machine as the benchmark driver (no remote endpoints).
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator

from agentperf.models import Framework, FrameworkConfig, Precision


class FrameworkLauncher(ABC):
    """Abstract base for framework-specific server process management.

    Subclasses implement build_command() and default_port; the shared
    lifecycle helpers (GPU validation, clock pinning, health polling,
    graceful shutdown) live here to avoid duplication.
    """

    @abstractmethod
    def build_command(self, config: FrameworkConfig) -> list[str]:
        """Return the argv list that will be handed to subprocess.Popen."""

    @property
    @abstractmethod
    def default_port(self) -> int:
        """Default HTTP port the server binds to."""

    def build_env(self, config: FrameworkConfig) -> dict[str, str] | None:
        """Return the environment mapping for the server process.

        Returns None to inherit the current process environment unchanged.
        Override to inject framework-specific variables (e.g. CUDA_VISIBLE_DEVICES).
        """
        return None

    def assert_gpu_free(self, gpu_ids: list[int], min_free_gb: float = 10.0) -> None:
        """Raise RuntimeError if any requested GPU lacks sufficient free VRAM.

        nvidia-smi reports free memory in MiB; we convert the caller-supplied
        threshold (in GB) to MiB once so the call site stays human-readable.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        min_free_mib = min_free_gb * 1024
        free_by_index: dict[int, float] = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                free_by_index[int(parts[0])] = float(parts[1])

        for gpu_id in gpu_ids:
            free_mib = free_by_index.get(gpu_id)
            if free_mib is None:
                raise RuntimeError(
                    f"GPU {gpu_id} not found in nvidia-smi output; "
                    "verify the GPU index is correct"
                )
            if free_mib < min_free_mib:
                free_gb = free_mib / 1024
                raise RuntimeError(
                    f"GPU {gpu_id} has only {free_gb:.1f} GB free VRAM "
                    f"but {min_free_gb} GB is required"
                )

    def lock_clocks(self, gpu_ids: list[int], freq_mhz: int) -> None:
        """Pin each GPU's graphics clock to freq_mhz for reproducible benchmarks.

        Clock variance is a primary source of run-to-run throughput jitter, so
        locking before a sweep gives more comparable numbers across runs.
        """
        for gpu in gpu_ids:
            subprocess.run(
                ["sudo", "nvidia-smi", "-i", str(gpu), "-lgc", str(freq_mhz)],
                check=True,
            )

    def unlock_clocks(self, gpu_ids: list[int]) -> None:
        """Restore automatic clock management on each GPU after a benchmark run."""
        for gpu in gpu_ids:
            subprocess.run(
                ["sudo", "nvidia-smi", "-i", str(gpu), "-rgc"],
                check=True,
            )

    @contextmanager
    def launch(
        self,
        config: FrameworkConfig,
        timeout_s: float = 120.0,
    ) -> Generator[subprocess.Popen, None, None]:
        """Start the server process and yield it once the /health endpoint responds.

        The health URL is derived from config.base_url so that non-default ports
        set by the caller are honoured automatically. On context exit the process
        receives SIGTERM; after a five-second grace period it is SIGKILLed if it
        has not yet terminated.
        """
        self.assert_gpu_free(config.gpu_ids)

        cmd = self.build_command(config)
        env = self.build_env(config)

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        health_url = config.base_url.rstrip("/") + "/health"
        deadline = time.monotonic() + timeout_s

        try:
            while True:
                # A premature exit before the server becomes healthy is always
                # a fatal error; capture it before sleeping so we fail fast.
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server process exited before becoming healthy "
                        f"(return code {process.returncode})"
                    )

                try:
                    with urllib.request.urlopen(health_url, timeout=2) as resp:
                        if resp.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    # Connection refused, server not yet listening — keep waiting.
                    pass

                if time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise TimeoutError(
                        f"Server did not respond to GET {health_url} "
                        f"within {timeout_s:.0f}s"
                    )

                time.sleep(1.0)

            yield process

        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Grace period elapsed; force-terminate to avoid orphan GPU processes.
                    process.kill()


class VLLMLauncher(FrameworkLauncher):
    """Launcher for the vLLM OpenAI-compatible API server."""

    # Class variable satisfies the abstract property; looked up first in MRO.
    default_port: int = 8000

    def build_command(self, config: FrameworkConfig) -> list[str]:
        cmd: list[str] = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
            "--port", "8000",
            "--tensor-parallel-size", str(len(config.gpu_ids)),
        ]

        # vLLM defaults to bf16 when no quantization flag is given, so BF16
        # precision requires no extra arguments.
        if config.precision == Precision.FP8:
            cmd += ["--quantization", "fp8"]
        elif config.precision == Precision.NVFP4:
            cmd += ["--quantization", "nvfp4"]

        # Caller-supplied overrides are appended last so they can supersede
        # any defaults constructed above.
        for key, value in config.extra_args.items():
            cmd += [f"--{key}", str(value)]

        return cmd

    def build_env(self, config: FrameworkConfig) -> dict[str, str]:
        """Restrict CUDA device visibility to the requested GPU indices.

        Passing gpu_ids=[2,3] maps them to CUDA ordinals 0 and 1 inside the
        subprocess, which is what vLLM's tensor-parallel machinery expects.
        """
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpu_ids)
        return env


class SGLangLauncher(FrameworkLauncher):
    """Launcher for the SGLang inference server."""

    default_port: int = 30000

    def build_command(self, config: FrameworkConfig) -> list[str]:
        return [
            "python", "-m", "sglang.launch_server",
            "--model-path", config.model,
            "--port", "30000",
            "--tp", str(len(config.gpu_ids)),
        ]


class TRTLLMLauncher(FrameworkLauncher):
    """Launcher for the TensorRT-LLM serving runtime."""

    default_port: int = 8000

    def build_command(self, config: FrameworkConfig) -> list[str]:
        return [
            "trtllm-serve", config.model,
            "--port", "8000",
            "--tp_size", str(len(config.gpu_ids)),
        ]


def get_launcher(framework: Framework) -> FrameworkLauncher:
    """Return an instantiated launcher for the given framework.

    Using a dict instead of if/elif keeps extension cost at O(1): add a new
    Framework enum value and a mapping entry, nothing else changes here.
    """
    registry: dict[Framework, type[FrameworkLauncher]] = {
        Framework.VLLM: VLLMLauncher,
        Framework.SGLANG: SGLangLauncher,
        Framework.TRTLLM: TRTLLMLauncher,
    }
    cls = registry.get(framework)
    if cls is None:
        raise ValueError(
            f"No launcher registered for framework {framework!r}. "
            f"Known frameworks: {list(registry)}"
        )
    return cls()
