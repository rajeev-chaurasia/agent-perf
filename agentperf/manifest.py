"""Creates, saves, and loads RunManifest.

Single responsibility: manifest lifecycle (create → save → load).
All I/O-bound public functions are async.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agentperf.models import (
    Framework,
    GpuSample,
    Precision,
    ReplayMode,
    RunManifest,
)

_DEFAULT_PACKAGES: list[str] = [
    "httpx",
    "pandas",
    "pyarrow",
    "numpy",
    "pydantic",
    "vllm",
    "sglang",
]


# ── Private helpers (synchronous) ──────────────────────────────────────────────

def _sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*, read in 64 KB chunks."""
    digest = hashlib.sha256()
    chunk_size = 64 * 1024  # 64 KB
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(repo_path: str | Path) -> str:
    """Return the short HEAD SHA of the git repo at *repo_path*.

    Returns ``"unknown"`` on any error (not a git repo, git not installed, …).
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_snapshot(package_names: list[str] | None = None) -> dict:
    """Return a snapshot of the Python environment.

    Returns::

        {
            "python": "3.11.x",
            "packages": {"httpx": "0.27.0", ...},
        }

    Packages that are not installed get the value ``"not-installed"``.
    """
    import sys

    names = package_names if package_names is not None else _DEFAULT_PACKAGES
    packages: dict[str, str] = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            packages[name] = "not-installed"

    return {
        "python": sys.version.split()[0],
        "packages": packages,
    }


# ── Public async API ──────────────────────────────────────────────────────────

async def create_manifest(
    *,
    framework: Framework,
    model: str,
    precision: Precision,
    gpu_ids: list[int],
    clock_mhz_locked: int,
    config_dict: dict,
    trace_path: str | Path,
    replay_mode: ReplayMode,
    concurrency: int,
    gpu_samples: list[GpuSample] | None = None,
    client_cpu_pct_mean: float = -1.0,
    repo_path: str | Path = ".",
) -> RunManifest:
    """Build and return a :class:`~agentperf.models.RunManifest`.

    Reads *trace_path* to extract ``trace_id`` and computes its SHA-256
    checksum.  Heavy file I/O is offloaded to a thread via
    :func:`asyncio.to_thread` so the event loop is not blocked.
    """
    trace_path = Path(trace_path)

    # Offload blocking I/O to a thread pool worker.
    trace_text: str = await asyncio.to_thread(trace_path.read_text, encoding="utf-8")
    trace_checksum: str = await asyncio.to_thread(_sha256_file, trace_path)

    trace_data = json.loads(trace_text)
    trace_id: str = trace_data["trace_id"]

    git_sha = await asyncio.to_thread(_git_sha, repo_path)
    env = await asyncio.to_thread(_env_snapshot)
    env["git_sha"] = git_sha

    run_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    return RunManifest(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        framework=framework,
        model=model,
        precision=precision,
        gpu_ids=gpu_ids,
        clock_mhz_locked=clock_mhz_locked,
        config_dict=config_dict,
        trace_id=trace_id,
        trace_checksum=trace_checksum,
        replay_mode=replay_mode,
        concurrency=concurrency,
        env_snapshot=env,
        client_cpu_pct_mean=client_cpu_pct_mean,
        gpu_samples=gpu_samples or [],
    )


async def save_manifest(manifest: RunManifest, output_dir: str | Path) -> Path:
    """Serialise *manifest* to ``<output_dir>/<run_id>.manifest.json``.

    Creates *output_dir* if it does not exist.  Returns the path of the
    written file.
    """
    output_dir = Path(output_dir)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

    dest = output_dir / f"{manifest.run_id}.manifest.json"
    payload = manifest.model_dump_json(indent=2)
    await asyncio.to_thread(dest.write_text, payload, "utf-8")
    return dest


async def load_manifest(path: str | Path) -> RunManifest:
    """Deserialise a :class:`~agentperf.models.RunManifest` from *path*."""
    text: str = await asyncio.to_thread(Path(path).read_text, "utf-8")
    return RunManifest.model_validate_json(text)
