#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Chapter 3 - vLLM scheduler knob sweep
# Fixed: vLLM, Llama-3.1-8B-Instruct FP8, GPU 1
# Each config changes exactly ONE knob vs base_8b_fp8.json
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTPERF_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRACES_DIR="${SCRIPT_DIR}/traces"
GPU_ID=1
# Override via environment: LOCK_MHZ=1755 ./run.sh
LOCK_MHZ="${LOCK_MHZ:-3090}"
REPLAY_CONCURRENCY="${REPLAY_CONCURRENCY:-8}"

# ---------------------------------------------------------------------------
# Ensure results dir exists before any file writes
# ---------------------------------------------------------------------------
mkdir -p "${SCRIPT_DIR}/results"

# ---------------------------------------------------------------------------
# Clock lock/unlock - lock BEFORE first server start, unlock on EXIT and ERR
# ---------------------------------------------------------------------------
_unlock_clocks() {
    echo "[trap] Restoring automatic clock management on GPU ${GPU_ID}"
    sudo nvidia-smi -i "${GPU_ID}" -rgc || true
}
trap '_unlock_clocks' EXIT ERR

echo "=== Locking GPU ${GPU_ID} graphics clock to ${LOCK_MHZ} MHz ==="
sudo nvidia-smi -i "${GPU_ID}" -lgc "${LOCK_MHZ}"

# ---------------------------------------------------------------------------
# Build traces if not already present
# ---------------------------------------------------------------------------
mkdir -p "${TRACES_DIR}"
for preset in agent_deep agent_swarm; do
    TRACE_FILE="${TRACES_DIR}/${preset}.json"
    if [[ ! -f "${TRACE_FILE}" ]]; then
        echo "Generating trace: ${preset} -> ${TRACE_FILE}"
        agentperf-generate --preset "${preset}" --output "${TRACE_FILE}"
    else
        echo "Trace already exists: ${TRACE_FILE}"
    fi
done

# ---------------------------------------------------------------------------
# Helper: poll /health until the server is ready (max 180 s)
# ---------------------------------------------------------------------------
_wait_healthy() {
    local base_url="$1"
    local url="${base_url%/}/health"
    local timeout_s=180
    echo "Polling ${url} (timeout ${timeout_s}s)..."
    for i in $(seq 1 "${timeout_s}"); do
        if curl -sf "${url}" > /dev/null 2>&1; then
            echo "Server healthy after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: server did not respond at ${url} within ${timeout_s}s" >&2
    return 1
}

# ---------------------------------------------------------------------------
# Helper: gracefully stop the server and allow GPU memory to drain
# ---------------------------------------------------------------------------
_stop_server() {
    local pid="$1"
    echo "Stopping server PID=${pid}"
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    sleep 5
}

# ---------------------------------------------------------------------------
# Helper: extract a scalar field or flattened extra_args from a config JSON.
#   Usage: _cfg <config_file> <field|extra_args_flat>
# ---------------------------------------------------------------------------
_cfg() {
    python3 - "$1" "$2" <<'PYEOF'
import json, sys

config_file, field = sys.argv[1], sys.argv[2]
with open(config_file) as fh:
    c = json.load(fh)

if field == "extra_args_flat":
    # One token per line so bash mapfile builds a clean array.
    # Keys already carry the "--" prefix as stored in the JSON.
    for k, v in c["extra_args"].items():
        print(k)
        if v:
            print(str(v))
else:
    print(c[field])
PYEOF
}

# ---------------------------------------------------------------------------
# Main sweep - iterate over every config in configs/
# ---------------------------------------------------------------------------
for config_file in "${SCRIPT_DIR}/configs/"*.json; do
    config_name="$(basename "${config_file}" .json)"

    echo ""
    echo "======================================================================"
    echo "=== Testing: ${config_name} ==="
    echo "======================================================================"

    MODEL=$(_cfg "${config_file}" model)
    BASE_URL=$(_cfg "${config_file}" base_url)

    # Build extra-args array; keys include "--" prefix already
    mapfile -t EXTRA_ARGS < <(_cfg "${config_file}" extra_args_flat)

    # Ensure per-config result directory exists before server start
    mkdir -p "${SCRIPT_DIR}/results/${config_name}"

    # --- Start vLLM server ---
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port 8000 \
        --tensor-parallel-size 1 \
        --quantization fp8 \
        "${EXTRA_ARGS[@]}" \
        > "${SCRIPT_DIR}/results/${config_name}/server.log" 2>&1 &
    SERVER_PID=$!
    echo "Server started (PID=${SERVER_PID})"

    if ! _wait_healthy "${BASE_URL}"; then
        echo "ERROR: aborting ${config_name}" >&2
        _stop_server "${SERVER_PID}"
        exit 1
    fi

    # --- Warmup: 60 seconds of traffic (results discarded) ---
    echo "--- Warmup: 60s of traffic ---"
    WARMUP_DEADLINE=$(( $(date +%s) + 60 ))
    while (( $(date +%s) < WARMUP_DEADLINE )); do
        REMAINING=$(( WARMUP_DEADLINE - $(date +%s) ))
        # timeout exits non-zero when the deadline fires; suppress that.
        timeout "${REMAINING}" agentperf-replay \
            --mode closed_loop \
            --concurrency "${REPLAY_CONCURRENCY}" \
            --trace "${TRACES_DIR}/agent_deep.json" \
            --base-url "${BASE_URL}" \
            --output-dir "${SCRIPT_DIR}/results/${config_name}/warmup" \
            --framework vllm \
            --model "${MODEL}" \
            --precision fp8 \
            --gpu-ids "${GPU_ID}" || true
    done
    echo "--- Warmup complete ---"

    # --- Measured runs ---
    for run in 1 2 3; do
        for trace in agent_deep agent_swarm; do
            echo "--- Run ${run} | trace=${trace} | config=${config_name} ---"
            agentperf-replay \
                --mode closed_loop \
                --concurrency "${REPLAY_CONCURRENCY}" \
                --trace "${TRACES_DIR}/${trace}.json" \
                --base-url "${BASE_URL}" \
                --output-dir "${SCRIPT_DIR}/results/${config_name}/run${run}_${trace}" \
                --framework vllm \
                --model "${MODEL}" \
                --precision fp8 \
                --gpu-ids "${GPU_ID}"
        done
    done

    _stop_server "${SERVER_PID}"
    echo "=== Finished: ${config_name} ==="
done

echo ""
echo "======================================================================"
echo "=== All configs complete. Results in ${SCRIPT_DIR}/results/ ==="
echo "======================================================================"
