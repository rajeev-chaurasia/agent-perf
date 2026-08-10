#!/usr/bin/env bash
set -euo pipefail

# Chapter 2: BF16 vs FP8 vs NVFP4 on 8B and 70B models.
# Measures throughput across the precision ladder using vLLM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIGS_DIR="${SCRIPT_DIR}/configs"
RESULTS_DIR="${SCRIPT_DIR}/results"
TRACE_DIR="${SCRIPT_DIR}/traces"
TRACE_PATH="${TRACE_DIR}/agent_deep.json"
QUALITY_TASK_PACK="${REPO_ROOT}/quality/evaluation-tasks.json"

WARMUP_SECONDS=60
CLOCK_MHZ=3090
CONCURRENCY=8
REPLAY_MODE="closed_loop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

gpu_ids_from_config() {
    python3 -c "
import json, sys
cfg = json.load(open('$1'))
print(','.join(str(g) for g in cfg['gpu_ids']))
"
}

lock_clocks() {
    local gpu_csv="$1"
    IFS=',' read -ra GPUS <<< "${gpu_csv}"
    for gpu in "${GPUS[@]}"; do
        sudo nvidia-smi -i "${gpu}" -lgc "${CLOCK_MHZ}"
    done
    echo "[clocks] Locked GPU(s) ${gpu_csv} to ${CLOCK_MHZ} MHz"
}

unlock_clocks() {
    local gpu_csv="$1"
    IFS=',' read -ra GPUS <<< "${gpu_csv}"
    for gpu in "${GPUS[@]}"; do
        sudo nvidia-smi -i "${gpu}" -rgc
    done
    echo "[clocks] Restored auto-boost on GPU(s) ${gpu_csv}"
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[server] Stopping PID ${SERVER_PID}"
        kill "${SERVER_PID}"
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}

wait_for_health() {
    local base_url="$1"
    local health_url="${base_url%/}/health"
    local deadline=$(( $(date +%s) + 300 ))
    echo "[health] Polling ${health_url} ..."
    until curl -sf "${health_url}" > /dev/null 2>&1; do
        if (( $(date +%s) > deadline )); then
            die "Server did not become healthy within 300s at ${health_url}"
        fi
        sleep 2
    done
    echo "[health] Server healthy at ${health_url}"
}

run_replay() {
    local config_file="$1"
    local output_subdir="$2"
    local label="$3"

    local framework model precision base_url
    framework=$(python3 -c "import json; c=json.load(open('${config_file}')); print(c['framework'])")
    model=$(python3 -c "import json; c=json.load(open('${config_file}')); print(c['model'])")
    precision=$(python3 -c "import json; c=json.load(open('${config_file}')); print(c['precision'])")
    base_url=$(python3 -c "import json; c=json.load(open('${config_file}')); print(c['base_url'])")

    mkdir -p "${output_subdir}"

    agentperf-replay \
        --mode "${REPLAY_MODE}" \
        --concurrency "${CONCURRENCY}" \
        --trace "${TRACE_PATH}" \
        --base-url "${base_url}" \
        --output-dir "${output_subdir}" \
        --framework "${framework}" \
        --model "${model}" \
        --precision "${precision}"

    echo "[replay] ${label} complete -> ${output_subdir}"
}

# ---------------------------------------------------------------------------
# Per-config benchmark function
# ---------------------------------------------------------------------------

run_config() {
    local config_file="$1"
    local config_name
    config_name="$(basename "${config_file}" .json)"

    local framework model precision base_url gpu_csv
    framework=$(python3 -c "import json; c=json.load(open('${config_file}')); print(c['framework'])")
    model=$(python3 -c "import json; c=json.load(open('${config_file}')); print(c['model'])")
    precision=$(python3 -c "import json; c=json.load(open('${config_file}')); print(c['precision'])")
    base_url=$(python3 -c "import json; c=json.load(open('${config_file}')); print(c['base_url'])")
    gpu_csv=$(gpu_ids_from_config "${config_file}")

    echo ""
    echo "========================================================"
    echo " Config: ${config_name}"
    echo " Model:  ${model}"
    echo " Prec:   ${precision}  |  GPUs: ${gpu_csv}"
    echo "========================================================"

    lock_clocks "${gpu_csv}"
    trap "unlock_clocks '${gpu_csv}'; stop_server" EXIT ERR

    CUDA_VISIBLE_DEVICES="${gpu_csv}" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --port 8000 \
        --tensor-parallel-size "$(echo "${gpu_csv}" | tr ',' '\n' | wc -l)" \
        $(
            if [[ "${precision}" == "fp8" ]];   then echo "--quantization fp8"; fi
            if [[ "${precision}" == "nvfp4" ]]; then echo "--quantization nvfp4"; fi
        ) \
        >> "${RESULTS_DIR}/${config_name}_server.log" 2>&1 &
    SERVER_PID=$!

    wait_for_health "${base_url}"

    echo "[warmup] Running ${WARMUP_SECONDS}s warmup ..."
    local warmup_dir
    warmup_dir="$(mktemp -d)"
    timeout "${WARMUP_SECONDS}" agentperf-replay \
        --mode "${REPLAY_MODE}" \
        --concurrency "${CONCURRENCY}" \
        --trace "${TRACE_PATH}" \
        --base-url "${base_url}" \
        --output-dir "${warmup_dir}" \
        --framework "${framework}" \
        --model "${model}" \
        --precision "${precision}" \
        || true
    rm -rf "${warmup_dir}"
    echo "[warmup] Done."

    for run in 1 2 3; do
        echo "[run ${run}/3] Measuring ${config_name} ..."
        local run_dir="${RESULTS_DIR}/${config_name}/run${run}"
        mkdir -p "${run_dir}"
        run_replay "${config_file}" "${run_dir}" "${config_name}/run${run}"
    done

    # Quality scoring requires a responses JSONL produced by a separate
    # model-output collection step (not yet implemented in this script).
    # When ready:
    #   agentperf-score --task-pack "${QUALITY_TASK_PACK}" \
    #       --responses "${RESULTS_DIR}/${config_name}/responses.jsonl" \
    #       --output "${RESULTS_DIR}/${config_name}/score.json"

    stop_server
    unlock_clocks "${gpu_csv}"
    trap - EXIT ERR

    echo "[done] ${config_name} complete."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

mkdir -p "${RESULTS_DIR}" "${TRACE_DIR}"

if [[ ! -f "${TRACE_PATH}" ]]; then
    echo "[trace] Generating agent_deep trace ..."
    agentperf-generate --preset agent_deep --output "${TRACE_PATH}"
fi

CONFIGS=(
    "${CONFIGS_DIR}/8b_bf16.json"
    "${CONFIGS_DIR}/8b_fp8.json"
    "${CONFIGS_DIR}/8b_nvfp4.json"
    "${CONFIGS_DIR}/70b_bf16_tp2.json"
    "${CONFIGS_DIR}/70b_fp8.json"
    "${CONFIGS_DIR}/70b_nvfp4.json"
)

for config in "${CONFIGS[@]}"; do
    [[ -f "${config}" ]] || die "Config not found: ${config}"
    run_config "${config}"
done

echo ""
echo "All configs complete. Results in: ${RESULTS_DIR}"
