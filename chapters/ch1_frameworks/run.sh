#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# Pin GPU 1 to its max stable graphics clock for reproducible benchmarks.
# Find the right value first:
#   nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits -i 1
BASE_CLOCK=1980   # TODO: fill in after checking nvidia-smi
GPU_ID=1
MODEL="meta-llama/Llama-3.1-8B-Instruct"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRACES=(baseline_chat agent_shallow agent_deep agent_swarm)
FRAMEWORKS=(vllm sglang trtllm)

# ── Clock lock (before first server start) ────────────────────────────────────
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  sudo nvidia-smi -rgc
}
trap cleanup EXIT ERR

sudo nvidia-smi -i "${GPU_ID}" -lgc "${BASE_CLOCK}"

# ── Pre-flight: build traces (idempotent) ─────────────────────────────────────
mkdir -p "${SCRIPT_DIR}/traces"

for preset in agent_shallow agent_deep agent_swarm; do
  if [[ ! -f "${SCRIPT_DIR}/traces/${preset}.json" ]]; then
    agentperf-generate --preset "${preset}" --output "${SCRIPT_DIR}/traces/${preset}.json"
  fi
done

# baseline_chat requires a ShareGPT source file; build it separately with:
#   python "${REPO_ROOT}/traces/baseline_chat/build.py" \
#     --input /path/to/ShareGPT_Vicuna_unfiltered.json \
#     --output "${SCRIPT_DIR}/traces/baseline_chat.json"
if [[ ! -f "${SCRIPT_DIR}/traces/baseline_chat.json" ]]; then
  echo "ERROR: ${SCRIPT_DIR}/traces/baseline_chat.json is missing." >&2
  echo "Build it from ShareGPT data before running this script." >&2
  exit 1
fi

mkdir -p results/

# ── Main sweep ────────────────────────────────────────────────────────────────
for FRAMEWORK in "${FRAMEWORKS[@]}"; do
  case "${FRAMEWORK}" in
    vllm)   BASE_URL="http://localhost:8000"  ;;
    sglang) BASE_URL="http://localhost:30000" ;;
    trtllm) BASE_URL="http://localhost:8000"  ;;
  esac

  # Start framework server
  case "${FRAMEWORK}" in
    vllm)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port 8000 \
        --tensor-parallel-size 1 \
        --enable-prefix-caching \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 &
      SERVER_PID=$!
      ;;
    sglang)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --port 30000 \
        --tp 1 \
        --mem-fraction-static 0.85 \
        --context-length 32768 &
      SERVER_PID=$!
      ;;
    trtllm)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" trtllm-serve "${MODEL}" \
        --port 8000 \
        --tp_size 1 \
        --kv_cache_free_gpu_mem_fraction 0.85 &
      SERVER_PID=$!
      ;;
  esac

  # Wait for /health
  echo "[ch1] Waiting for ${FRAMEWORK} /health ..."
  until curl -sf "${BASE_URL}/health" > /dev/null 2>&1; do sleep 2; done
  echo "[ch1] ${FRAMEWORK} ready."

  for TRACE in "${TRACES[@]}"; do
    TRACE_PATH="${SCRIPT_DIR}/traces/${TRACE}.json"

    # Warmup: 60 seconds of closed-loop traffic before the measured window.
    # timeout exits 124 on expiry, which is expected — suppress with || true.
    timeout 60 agentperf-replay \
      --mode closed_loop \
      --concurrency 8 \
      --trace "${TRACE_PATH}" \
      --base-url "${BASE_URL}" \
      --framework "${FRAMEWORK}" \
      --model "${MODEL}" \
      --precision bf16 \
      --gpu-ids "${GPU_ID}" \
      --output-dir "results/${FRAMEWORK}/${TRACE}/warmup" || true

    for run in 1 2 3; do
      for CONCURRENCY in 1 2 4 8 16 32 64 128; do
        agentperf-replay \
          --mode closed_loop \
          --concurrency "${CONCURRENCY}" \
          --trace "${TRACE_PATH}" \
          --base-url "${BASE_URL}" \
          --framework "${FRAMEWORK}" \
          --model "${MODEL}" \
          --precision bf16 \
          --gpu-ids "${GPU_ID}" \
          --output-dir "results/${FRAMEWORK}/${TRACE}/run${run}"
      done
    done
  done

  # Stop server before launching the next framework
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
done

# ── Report ────────────────────────────────────────────────────────────────────
mapfile -t PARQUET_FILES < <(
  find results -name "*.parquet" ! -path "*/warmup/*" | sort
)

PARQUET_ARGS=()
LABEL_ARGS=()
for f in "${PARQUET_FILES[@]}"; do
  PARQUET_ARGS+=("${f}")
  # Derive label from path: results/<fw>/<trace>/run<N>/<run_id>.parquet
  #   -> <fw>/<trace>/run<N>
  label="$(echo "${f}" | sed 's|^results/||; s|/[^/]*\.parquet$||')"
  LABEL_ARGS+=("${label}")
done

agentperf-report \
  --parquet "${PARQUET_ARGS[@]}" \
  --labels  "${LABEL_ARGS[@]}" \
  --output-dir results/report \
  --plot all
