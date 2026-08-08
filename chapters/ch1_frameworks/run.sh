#!/usr/bin/env bash
set -euo pipefail

BASE_CLOCK=3090
GPU_ID=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

# Separate venvs per framework: vllm torch==2.11.0, sglang torch==2.9.1, trtllm torch==2.11.0
PYTHON_VLLM="${REPO_ROOT}/.venv-vllm/bin/python"
PYTHON_SGLANG="${REPO_ROOT}/.venv-sglang/bin/python"
PYTHON_TRTLLM="${REPO_ROOT}/.venv-trtllm/bin/python"

MODEL="${REPO_ROOT}/models/llama-3.1-8b-instruct"

TRACES=(baseline_chat agent_shallow agent_deep agent_swarm)
FRAMEWORKS=(vllm sglang trtllm)
CONCURRENCY_LEVELS=(1 2 4 8 16 32 64 128)
RUNS=(1 2 3)

measurement_done() {
  local fw="$1" trace="$2" run="$3" concurrency="$4"
  local dir="${RESULTS_DIR}/${fw}/${trace}/run${run}/c${concurrency}"
  ls "${dir}"/*.parquet 2>/dev/null | grep -q .
}

framework_done_count() {
  local fw="$1"
  local count=0
  for t in "${TRACES[@]}"; do
    for r in "${RUNS[@]}"; do
      for c in "${CONCURRENCY_LEVELS[@]}"; do
        measurement_done "${fw}" "${t}" "${r}" "${c}" && (( count++ )) || true
      done
    done
  done
  echo "${count}"
}

total_measurements() {
  echo $(( ${#TRACES[@]} * ${#RUNS[@]} * ${#CONCURRENCY_LEVELS[@]} ))
}

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    # setsid creates a process group; kill the whole group to catch child workers
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  sudo nvidia-smi -rgc
}
trap cleanup EXIT ERR

sudo nvidia-smi -i "${GPU_ID}" -lgc "${BASE_CLOCK}"

mkdir -p "${SCRIPT_DIR}/traces"

for preset in agent_shallow agent_deep agent_swarm; do
  if [[ ! -f "${SCRIPT_DIR}/traces/${preset}.json" ]]; then
    agentperf-generate --preset "${preset}" --output "${SCRIPT_DIR}/traces/${preset}.json"
  fi
done

# baseline_chat needs a ShareGPT source file; build separately:
#   python traces/baseline_chat/build.py \
#     --input /path/to/ShareGPT_Vicuna_unfiltered.json \
#     --output chapters/ch1_frameworks/traces/baseline_chat.json
if [[ ! -f "${SCRIPT_DIR}/traces/baseline_chat.json" ]]; then
  echo "[ch1] baseline_chat.json not found - skipping that trace." >&2
  TRACES=(agent_shallow agent_deep agent_swarm)
fi

mkdir -p "${RESULTS_DIR}"

TOTAL=$(total_measurements)

for FRAMEWORK in "${FRAMEWORKS[@]}"; do
  case "${FRAMEWORK}" in
    vllm)   BASE_URL="http://localhost:8000";  MODEL_ID="${MODEL}"          ;;
    sglang) BASE_URL="http://localhost:30000"; MODEL_ID="${MODEL}"          ;;
    trtllm) BASE_URL="http://localhost:8000";  MODEL_ID="$(basename "${MODEL}")" ;;
  esac

  DONE=$(framework_done_count "${FRAMEWORK}")
  if [[ "${DONE}" -ge "${TOTAL}" ]]; then
    echo "[ch1] ${FRAMEWORK} fully complete (${DONE}/${TOTAL}) - skipping server start"
    continue
  fi
  echo "[ch1] ${FRAMEWORK}: ${DONE}/${TOTAL} measurements done - starting server"

  case "${FRAMEWORK}" in
    vllm)
      setsid env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_VLLM}" \
        -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_ID}" \
        --port 8000 \
        --tensor-parallel-size 1 \
        --enable-prefix-caching \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 &
      SERVER_PID=$!
      ;;
    sglang)
      setsid env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_SGLANG}" \
        -m sglang.launch_server \
        --model-path "${MODEL}" \
        --port 30000 \
        --tp 1 \
        --mem-fraction-static 0.85 \
        --context-length 32768 &
      SERVER_PID=$!
      ;;
    trtllm)
      setsid env CUDA_VISIBLE_DEVICES="${GPU_ID}" PATH="${REPO_ROOT}/.venv-trtllm/bin:${PATH}" \
        "${REPO_ROOT}/.venv-trtllm/bin/trtllm-serve" "${MODEL}" \
        --port 8000 \
        --tp_size 1 \
        --kv_cache_free_gpu_memory_fraction 0.85 \
        --max_num_tokens 32768 \
        --max_seq_len 32768 &
      SERVER_PID=$!
      ;;
  esac

  echo "[ch1] Waiting for ${FRAMEWORK} /health ..."
  until curl -sf "${BASE_URL}/health" > /dev/null 2>&1; do sleep 2; done
  echo "[ch1] ${FRAMEWORK} ready."

  for TRACE in "${TRACES[@]}"; do
    TRACE_PATH="${SCRIPT_DIR}/traces/${TRACE}.json"

    TRACE_DONE=0
    for r in "${RUNS[@]}"; do
      for c in "${CONCURRENCY_LEVELS[@]}"; do
        measurement_done "${FRAMEWORK}" "${TRACE}" "${r}" "${c}" && (( TRACE_DONE++ )) || true
      done
    done
    TRACE_TOTAL=$(( ${#RUNS[@]} * ${#CONCURRENCY_LEVELS[@]} ))

    if [[ "${TRACE_DONE}" -ge "${TRACE_TOTAL}" ]]; then
      echo "[ch1] ${FRAMEWORK}/${TRACE} fully done (${TRACE_DONE}/${TRACE_TOTAL}) - skipping"
      continue
    fi

    # 60s warmup on every server start to prime the prefix cache
    echo "[ch1] Warming up ${FRAMEWORK}/${TRACE} ..."
    timeout 60 agentperf-replay \
      --mode closed_loop \
      --concurrency 8 \
      --trace "${TRACE_PATH}" \
      --base-url "${BASE_URL}" \
      --framework "${FRAMEWORK}" \
      --model "${MODEL_ID}" \
      --precision bf16 \
      --gpu-ids "${GPU_ID}" \
      --output-dir "${RESULTS_DIR}/${FRAMEWORK}/${TRACE}/warmup" || true

    for run in "${RUNS[@]}"; do
      for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do
        OUT_DIR="${RESULTS_DIR}/${FRAMEWORK}/${TRACE}/run${run}/c${CONCURRENCY}"

        if measurement_done "${FRAMEWORK}" "${TRACE}" "${run}" "${CONCURRENCY}"; then
          echo "[ch1] skip ${FRAMEWORK}/${TRACE}/run${run}/c${CONCURRENCY} (already done)"
          continue
        fi

        echo "[ch1] measuring ${FRAMEWORK}/${TRACE}/run${run}/c${CONCURRENCY} ..."
        agentperf-replay \
          --mode closed_loop \
          --concurrency "${CONCURRENCY}" \
          --trace "${TRACE_PATH}" \
          --base-url "${BASE_URL}" \
          --framework "${FRAMEWORK}" \
          --model "${MODEL_ID}" \
          --precision bf16 \
          --gpu-ids "${GPU_ID}" \
          --output-dir "${OUT_DIR}"
      done
    done
  done

  kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
done

mapfile -t PARQUET_FILES < <(
  find "${RESULTS_DIR}" -name "*.parquet" ! -path "*/warmup/*" | sort
)

if [[ "${#PARQUET_FILES[@]}" -eq 0 ]]; then
  echo "[ch1] No result files found - nothing to report."
  exit 0
fi

PARQUET_ARGS=()
LABEL_ARGS=()
for f in "${PARQUET_FILES[@]}"; do
  PARQUET_ARGS+=("${f}")
  label="$(echo "${f}" | sed "s|^${RESULTS_DIR}/||; s|/[^/]*\.parquet\$||")"
  LABEL_ARGS+=("${label}")
done

agentperf-report \
  --parquet "${PARQUET_ARGS[@]}" \
  --labels  "${LABEL_ARGS[@]}" \
  --output-dir "${RESULTS_DIR}/report" \
  --plot all
