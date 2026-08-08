#!/usr/bin/env bash
set -euo pipefail

# --- Configurable knobs ---
# BASE_CLOCK: application clock to lock in MHz for reproducible benchmarks.
# Query available clocks with:
#   nvidia-smi --query-supported-clocks=gr --format=csv,noheader -i 0
# Owner must verify the correct base clock for the installed GPUs before running.
BASE_CLOCK="${BASE_CLOCK:-}"
if [[ -z "${BASE_CLOCK}" ]]; then
    echo "ERROR: BASE_CLOCK is not set. Export it before running:" >&2
    echo "  export BASE_CLOCK=<MHz>  # e.g. 3090 for 96 GB workstation GPU" >&2
    exit 1
fi

MODEL="models/llama-3.3-70b-instruct"
WARMUP_S=60

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
TRACE="${RESULTS_DIR}/agent_deep_trace.json"

# --- Cleanup: reset graphics clocks on any exit or error ---
# Must be registered before the first nvidia-smi -lgc call.
trap "sudo nvidia-smi -rgc; echo 'INFO: graphics clocks reset.'" EXIT ERR

# --- Prepare results directory ---
mkdir -p "${RESULTS_DIR}"

# --- NCCL all-reduce bandwidth microbenchmark (PCIe Gen5 P2P baseline) ---
# Run once before server tests so server traffic does not interfere.
echo "==> Running NCCL all-reduce bandwidth sweep (GPUs 0,1) ..."
"${REPO_ROOT}/nccl-tests/build/all_reduce_perf" \
    -b 1K -e 1G -f 2 -g 2 \
    | tee "${RESULTS_DIR}/nccl_bandwidth.txt"

# --- Generate agent_deep trace ---
if [[ ! -f "${TRACE}" ]]; then
    echo "==> Generating agent_deep trace ..."
    agentperf-generate --preset agent_deep --output "${TRACE}"
fi

# --- Lock graphics clocks on both GPUs ---
echo "==> Locking graphics clocks on GPUs 0 and 1 to ${BASE_CLOCK} MHz ..."
sudo nvidia-smi -i 0 -lgc "${BASE_CLOCK}"
sudo nvidia-smi -i 1 -lgc "${BASE_CLOCK}"

# --- Helper: wait for vLLM health ---
wait_for_server() {
    local url="$1"
    echo "    Waiting for server at ${url}/health ..."
    until curl -sf "${url}/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "    Server is healthy."
}

# --- Helper: 60-second warmup ---
warmup_server() {
    local label="$1"; shift
    local warmup_end=$(( $(date +%s) + WARMUP_S ))
    echo "==> Warming up ${label} for ${WARMUP_S}s ..."
    while [[ $(date +%s) -lt ${warmup_end} ]]; do
        agentperf-replay "$@" \
            --output-dir "${RESULTS_DIR}/warmup_${label}" \
            > /dev/null 2>&1 || true
    done
    echo "    Warmup complete."
}

# --- Helper: measured sweep (3 runs x 4 concurrency levels) ---
measured_sweep() {
    local label="$1"; shift
    echo "==> Measured sweep for ${label} ..."
    for run in 1 2 3; do
        for CONCURRENCY in 1 4 16 64; do
            echo "    run=${run}  concurrency=${CONCURRENCY}"
            agentperf-replay "$@" \
                --mode closed_loop \
                --concurrency "${CONCURRENCY}" \
                --trace "${TRACE}" \
                --output-dir "${RESULTS_DIR}/${label}/run${run}"
        done
    done
}
# Config 1 (base): 70B FP8 on single GPU (GPU 1)
#   Precision: fp8   GPU count: 1   Extra flags: none

echo ""
echo ">>> [1/2] Starting vLLM: 70B FP8 single GPU (GPU 1) ..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port 8000 \
    --quantization fp8 \
    &
SERVER_PID=$!

wait_for_server "http://localhost:8000"

warmup_server "fp8_single" \
    --framework vllm \
    --model "${MODEL}" \
    --precision fp8 \
    --gpu-ids 1 \
    --mode closed_loop \
    --concurrency 4 \
    --trace "${TRACE}" \
    --base-url "http://localhost:8000"

measured_sweep "fp8_single" \
    --framework vllm \
    --model "${MODEL}" \
    --precision fp8 \
    --gpu-ids 1 \
    --base-url "http://localhost:8000"

echo "    Stopping fp8_single server ..."
kill "${SERVER_PID}" && wait "${SERVER_PID}" 2>/dev/null || true
sleep 5
# Config 2 (comparison): 70B BF16 TP=2 on GPUs 0,1 over PCIe Gen5 (no NVLink)
#   Precision: bf16   GPU count: 2   Extra flags: --tensor-parallel-size 2
#   ONE variable changes vs base: precision+parallelism strategy

echo ""
echo ">>> [2/2] Starting vLLM: 70B BF16 TP=2 (GPUs 0,1, PCIe Gen5, no NVLink) ..."
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port 8000 \
    --tensor-parallel-size 2 \
    &
SERVER_PID=$!

wait_for_server "http://localhost:8000"

warmup_server "bf16_tp2" \
    --framework vllm \
    --model "${MODEL}" \
    --precision bf16 \
    --gpu-ids 0,1 \
    --mode closed_loop \
    --concurrency 4 \
    --trace "${TRACE}" \
    --base-url "http://localhost:8000"

measured_sweep "bf16_tp2" \
    --framework vllm \
    --model "${MODEL}" \
    --precision bf16 \
    --gpu-ids 0,1 \
    --base-url "http://localhost:8000"

echo "    Stopping bf16_tp2 server ..."
kill "${SERVER_PID}" && wait "${SERVER_PID}" 2>/dev/null || true

# --- Optional: nsys trace for one serving window (uncomment to enable) ---
# Profiles the bf16_tp2 config at concurrency=16 to capture CUDA and NVTX events.
# nsys profile --trace=cuda,nvtx \
#     --output="${RESULTS_DIR}/nsys_trace" \
#     agentperf-replay \
#         --framework vllm \
#         --model "${MODEL}" \
#         --precision bf16 \
#         --gpu-ids 0,1 \
#         --mode closed_loop \
#         --concurrency 16 \
#         --trace "${TRACE}" \
#         --base-url "http://localhost:8000" \
#         --output-dir "${RESULTS_DIR}/nsys_run"

echo ""
echo "==> Chapter 4 run complete. Results written to: ${RESULTS_DIR}"
