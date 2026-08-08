"""Central data models and enums for agentperf.
Single source of truth. Every other module imports from here.
"""
from __future__ import annotations
import enum
from pydantic import BaseModel, Field, model_validator


# --- Enums

class Framework(str, enum.Enum):
    """Supported inference serving frameworks."""
    VLLM = "vllm"
    SGLANG = "sglang"
    TRTLLM = "trtllm"

class Precision(str, enum.Enum):
    """Model weight precision / quantization format."""
    BF16 = "bf16"
    FP8 = "fp8"
    NVFP4 = "nvfp4"

class ReplayMode(str, enum.Enum):
    """Trace replay concurrency model."""
    CLOSED_LOOP = "closed_loop"
    OPEN_LOOP = "open_loop"

class RoleSequence(str, enum.Enum):
    USER = "user"
    TOOL_RESULT = "tool_result"

class CheckType(str, enum.Enum):
    CODE_EXEC = "code_exec"
    FUNCTION_CALL = "function_call"
    EXACT_MATCH = "exact_match"

class QAStatus(str, enum.Enum):
    PASS_ = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


# --- Trace schema models

class SamplingConfig(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0

class TurnSpec(BaseModel):
    turn_id: int
    role_sequence: RoleSequence
    content_ref: str           # inline text or "@path/to/file.txt"
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    expects_tool_call: bool = False
    think_time_ms: int = 0

class SessionSpec(BaseModel):
    session_id: str
    system_prompt_ref: str
    turns: list[TurnSpec]

class TraceSpec(BaseModel):
    trace_id: str
    schema_version: int = 1
    sessions: list[SessionSpec]

    @model_validator(mode="after")
    def check_version(self) -> "TraceSpec":
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        return self


# --- Measurement models

class TurnResult(BaseModel):
    """Per-request measurement. All ns timestamps from time.monotonic_ns()."""
    session_id: str
    turn_id: int
    request_sent_ns: int
    first_token_ns: int = 0       # 0 = no first token (error or not streaming)
    last_token_ns: int = 0        # 0 = stream did not complete
    prompt_tokens: int = -1       # -1 = not reported by server
    output_tokens: int = -1
    http_status: int
    cache_hit: bool | None = None
    run_id: str

class GpuSample(BaseModel):
    ts_ms: float
    gpu_id: int
    util_pct: float
    power_w: float
    vram_used_gb: float

class RunManifest(BaseModel):
    run_id: str
    timestamp_utc: str
    framework: Framework
    model: str
    precision: Precision
    gpu_ids: list[int]
    clock_mhz_locked: int = 0
    config_dict: dict
    trace_id: str
    trace_checksum: str
    replay_mode: ReplayMode
    concurrency: int
    env_snapshot: dict
    client_cpu_pct_mean: float = -1.0
    gpu_samples: list[GpuSample] = Field(default_factory=list)


# --- Config models

class FrameworkConfig(BaseModel):
    framework: Framework
    model: str
    precision: Precision
    gpu_ids: list[int] = Field(default_factory=lambda: [1])  # GPU 1 = non-display
    base_url: str = "http://localhost:8000"
    extra_args: dict = Field(default_factory=dict)

class ReplayConfig(BaseModel):
    mode: ReplayMode
    concurrency: int
    trace_path: str
    base_url: str = "http://localhost:8000"
    output_dir: str = "results"
    run_id: str | None = None
    model_name: str = ""


# --- Quality models

class TaskItem(BaseModel):
    id: str
    category: str
    prompt: str
    expected_output: str
    check_type: CheckType
    unit_tests: str | None = None

class ScoreResult(BaseModel):
    task_id: str
    passed: bool
    error: str | None = None
