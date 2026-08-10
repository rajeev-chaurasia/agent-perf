# Setup

## Hardware Requirements

The benchmarks in this repo target a single 96 GB GPU (sm_120) on PCIe Gen5. Chapter 1 runs on GPU 1 exclusively, leaving GPU 0 free for display. CUDA 13.2 and driver 595.84 or later are required. The host used for development is an Intel Core Ultra 9 285K with 125 GB RAM running Ubuntu 24.04.

Chapters that test tensor parallelism (TP=2) additionally require GPU 0 at the same clock settings.

If you are on a different sm_120 SKU, confirm that Flash Attention 3 or FlashInfer is available for your sm_120 variant before running vLLM. See `ENVIRONMENTS.md` for the known sm_120 issue list.

## Base Environment

Python 3.12.3 is the tested interpreter. The agentperf harness and all analysis code go into a single base venv at `.venv`:

```bash
pip install -e ".[dev]"
```

This installs the `agentperf-replay`, `agentperf-report`, and `agentperf-generate` CLI tools, plus test dependencies (pytest, pytest-asyncio, ruff).

## Framework Virtual Environments

Each inference framework requires its own virtual environment because vLLM, SGLang, and TRT-LLM each pin incompatible PyTorch versions. The harness itself does not import any of them; it only talks to them over HTTP.

**vLLM (.venv-vllm)**

```bash
uv venv .venv-vllm
uv pip install vllm==0.26.0 --python .venv-vllm/bin/python
```

vLLM 0.26.0 ships a pre-built wheel for sm_120 on CUDA 12.8+. If JIT compilation issues appear with the FlashInfer sampler (GitHub vLLM #50747), install the pre-compiled FlashInfer wheel for sm_120 before starting the server.

**SGLang (.venv-sglang)**

```bash
uv venv .venv-sglang
uv pip install "sglang[all]==0.5.9" --python .venv-sglang/bin/python
```

The `[all]` extra pulls in FlashInfer, the Triton kernels, and the radix cache backend. SGLang listens on port 30000 by default.

**TensorRT-LLM (.venv-trtllm)**

```bash
uv venv .venv-trtllm
uv pip install tensorrt-llm==1.3.0rc23 ninja --python .venv-trtllm/bin/python
```

`ninja` is required for the JIT-compiled TRT-LLM CUDA extensions. TRT-LLM uses `trtllm-serve` as its OpenAI-compatible frontend, which installs into `.venv-trtllm/bin/`.

## Model Download

Download the base model into the `models/` directory:

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir models/llama-3.1-8b-instruct
```

The expected revision is `0e9e39f249a16976918f6564b8830bc894c89659`. Verify after download if reproducibility matters. Quantized checkpoints (FP8, NVFP4) are generated separately using the scripts in `quantize/`; see `ENVIRONMENTS.md` for the full model and checkpoint table.

## GPU Clock Lock

Lock the graphics clock before starting any measurement. Without this, boost clock variation adds noise to latency measurements, particularly ITL at high concurrency.

```bash
sudo nvidia-smi -i 1 -lgc 3090
```

The max stable graphics clock on the benchmark GPU is 3090 MHz. The `run.sh` scripts call this automatically and register a cleanup trap to release the lock (`sudo nvidia-smi -rgc`) on exit or error.

## Smoke Test

To verify the harness without a GPU, start a mock server and run a short replay against it:

```bash
# Terminal 1: start a minimal mock OpenAI-compatible server
uvicorn agentperf.launch:app --port 8000

# Terminal 2: run two sessions at concurrency 2
agentperf-replay \
    --mode closed_loop \
    --concurrency 2 \
    --trace chapters/ch1_frameworks/traces/agent_shallow.json \
    --base-url http://localhost:8000 \
    --framework vllm \
    --model test-model \
    --precision bf16 \
    --output-dir /tmp/smoke
```

The mock server returns short, fixed-length responses. TTFT and ITL values will be artificially low, but the Parquet output and manifest should be written cleanly to `/tmp/smoke/`.

## Running a Chapter

Each chapter has a self-contained `run.sh` that handles server lifecycle, clock locking, trace generation, the full concurrency sweep, and report generation:

```bash
bash chapters/ch1_frameworks/run.sh
```

The script is idempotent. It checks for an existing `.parquet` file before each measurement and skips ahead if one is found, so an interrupted run can be resumed by re-running the same command. Results land in `chapters/ch1_frameworks/results/` and the report (plots + summary table) in `chapters/ch1_frameworks/results/report/`.
